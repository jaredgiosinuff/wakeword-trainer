const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, ScanCommand, GetCommand, PutCommand, UpdateCommand, QueryCommand } = require('@aws-sdk/lib-dynamodb');
const { S3Client, PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const crypto = require('crypto');

const ddbClient = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(ddbClient);
const s3Client = new S3Client({});

const WAKEWORDS_TABLE = process.env.WAKEWORDS_TABLE;
const SAMPLES_BUCKET = process.env.SAMPLES_BUCKET;

// Bayesian weighted rating constants
const MINIMUM_VOTES = 5; // Minimum votes before rating stabilizes
const PRIOR_MEAN = 3.0;  // Prior mean rating (neutral)

/**
 * Calculate Bayesian weighted rating
 * This prevents items with few ratings from dominating
 * Formula: (v / (v + m)) * R + (m / (v + m)) * C
 * where:
 *   R = average rating for this item
 *   v = number of votes for this item
 *   m = minimum votes required (MINIMUM_VOTES)
 *   C = prior mean rating (PRIOR_MEAN)
 */
function calculateWeightedRating(averageRating, voteCount) {
  if (voteCount === 0) return PRIOR_MEAN;

  const weight = voteCount / (voteCount + MINIMUM_VOTES);
  return (weight * averageRating) + ((1 - weight) * PRIOR_MEAN);
}

// CORS headers
const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
};

/**
 * List wakewords with search, sort, and pagination
 */
exports.list = async (event) => {
  try {
    const params = event.queryStringParameters || {};
    const search = params.search?.toLowerCase();
    const sortBy = params.sortBy || 'weightedRating'; // weightedRating, sampleCount, name, createdAt
    const sortOrder = params.sortOrder || 'desc';
    const limit = parseInt(params.limit) || 50;

    // Scan all wakewords (for small datasets, this is fine)
    // For larger datasets, use GSI with pagination
    const result = await docClient.send(new ScanCommand({
      TableName: WAKEWORDS_TABLE,
    }));

    let items = result.Items || [];

    // Filter by search term
    if (search) {
      items = items.filter(item =>
        item.name.toLowerCase().includes(search) ||
        (item.description && item.description.toLowerCase().includes(search))
      );
    }

    // Sort
    items.sort((a, b) => {
      let aVal = a[sortBy] || 0;
      let bVal = b[sortBy] || 0;

      if (sortBy === 'name') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
        return sortOrder === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });

    // Limit results
    items = items.slice(0, limit);

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        wakewords: items,
        count: items.length,
      }),
    };
  } catch (error) {
    console.error('Error listing wakewords:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to list wakewords' }),
    };
  }
};

/**
 * Get a single wakeword with sample URLs
 */
exports.get = async (event) => {
  try {
    const { id } = event.pathParameters;

    const result = await docClient.send(new GetCommand({
      TableName: WAKEWORDS_TABLE,
      Key: { id },
    }));

    if (!result.Item) {
      return {
        statusCode: 404,
        headers,
        body: JSON.stringify({ error: 'Wakeword not found' }),
      };
    }

    const wakeword = result.Item;

    // Generate presigned URLs for samples
    if (wakeword.samples && wakeword.samples.length > 0) {
      const sampleUrls = await Promise.all(
        wakeword.samples.map(async (sample) => {
          const url = await getSignedUrl(
            s3Client,
            new GetObjectCommand({
              Bucket: SAMPLES_BUCKET,
              Key: sample.s3Key,
            }),
            { expiresIn: 3600 }
          );
          return { ...sample, url };
        })
      );
      wakeword.samples = sampleUrls;
    }

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(wakeword),
    };
  } catch (error) {
    console.error('Error getting wakeword:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get wakeword' }),
    };
  }
};

/**
 * Create a new wakeword entry
 */
exports.create = async (event) => {
  try {
    const body = JSON.parse(event.body);
    const { name, description, agreedToShare } = body;

    if (!name || !agreedToShare) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          error: 'Name is required and you must agree to share the wakeword publicly'
        }),
      };
    }

    // Check if wakeword already exists
    const existing = await docClient.send(new QueryCommand({
      TableName: WAKEWORDS_TABLE,
      IndexName: 'name-index',
      KeyConditionExpression: '#name = :name',
      ExpressionAttributeNames: { '#name': 'name' },
      ExpressionAttributeValues: { ':name': name.toLowerCase() },
    }));

    if (existing.Items && existing.Items.length > 0) {
      // Return existing wakeword
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          wakeword: existing.Items[0],
          message: 'Wakeword already exists - you can add samples to it',
        }),
      };
    }

    const id = crypto.randomUUID();
    const now = new Date().toISOString();

    const wakeword = {
      id,
      name: name.toLowerCase(),
      displayName: name,
      description: description || '',
      samples: [],
      sampleCount: 0,
      ratingSum: 0,
      ratingCount: 0,
      averageRating: 0,
      weightedRating: PRIOR_MEAN,
      createdAt: now,
      updatedAt: now,
    };

    await docClient.send(new PutCommand({
      TableName: WAKEWORDS_TABLE,
      Item: wakeword,
    }));

    return {
      statusCode: 201,
      headers,
      body: JSON.stringify({ wakeword }),
    };
  } catch (error) {
    console.error('Error creating wakeword:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to create wakeword' }),
    };
  }
};
