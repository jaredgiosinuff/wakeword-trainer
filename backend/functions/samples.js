const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const crypto = require('crypto');

const ddbClient = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(ddbClient);
const s3Client = new S3Client({});

const WAKEWORDS_TABLE = process.env.WAKEWORDS_TABLE;
const SAMPLES_BUCKET = process.env.SAMPLES_BUCKET;

const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
};

/**
 * Get a presigned URL for uploading a sample
 * Client uploads directly to S3, then confirms upload
 */
exports.upload = async (event) => {
  try {
    const { id } = event.pathParameters;
    const body = JSON.parse(event.body);
    const { filename, contentType, duration, agreedToShare } = body;

    if (!agreedToShare) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          error: 'You must agree to share this sample publicly under CC0 license'
        }),
      };
    }

    // Verify wakeword exists
    const wakeword = await docClient.send(new GetCommand({
      TableName: WAKEWORDS_TABLE,
      Key: { id },
    }));

    if (!wakeword.Item) {
      return {
        statusCode: 404,
        headers,
        body: JSON.stringify({ error: 'Wakeword not found' }),
      };
    }

    const sampleId = crypto.randomUUID();
    const ext = filename?.split('.').pop() || 'webm';
    const s3Key = `wakewords/${id}/samples/${sampleId}.${ext}`;

    // Generate presigned URL for upload
    const uploadUrl = await getSignedUrl(
      s3Client,
      new PutObjectCommand({
        Bucket: SAMPLES_BUCKET,
        Key: s3Key,
        ContentType: contentType || 'audio/webm',
      }),
      { expiresIn: 300 } // 5 minutes to upload
    );

    // Add sample metadata to wakeword
    const sample = {
      id: sampleId,
      s3Key,
      duration: duration || 0,
      contentType: contentType || 'audio/webm',
      uploadedAt: new Date().toISOString(),
    };

    await docClient.send(new UpdateCommand({
      TableName: WAKEWORDS_TABLE,
      Key: { id },
      UpdateExpression: 'SET samples = list_append(if_not_exists(samples, :empty), :sample), sampleCount = if_not_exists(sampleCount, :zero) + :one, updatedAt = :now',
      ExpressionAttributeValues: {
        ':sample': [sample],
        ':empty': [],
        ':zero': 0,
        ':one': 1,
        ':now': new Date().toISOString(),
      },
    }));

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        uploadUrl,
        sampleId,
        s3Key,
        message: 'Upload your audio file to the provided URL',
      }),
    };
  } catch (error) {
    console.error('Error creating upload URL:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to create upload URL' }),
    };
  }
};
