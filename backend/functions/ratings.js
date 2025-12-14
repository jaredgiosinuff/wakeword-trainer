const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');

const ddbClient = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(ddbClient);

const WAKEWORDS_TABLE = process.env.WAKEWORDS_TABLE;
const RATINGS_TABLE = process.env.RATINGS_TABLE;

// Bayesian weighted rating constants
const MINIMUM_VOTES = 5; // Minimum votes before rating stabilizes
const PRIOR_MEAN = 3.0;  // Prior mean rating (neutral)

/**
 * Calculate Bayesian weighted rating
 *
 * This algorithm ensures that:
 * - A single 5-star rating doesn't outweigh 25 reviews at 4.5
 * - New items with few ratings are pulled toward the global mean
 * - As vote count increases, the weighted rating approaches the true average
 *
 * Formula: (v / (v + m)) * R + (m / (v + m)) * C
 * where:
 *   R = average rating for this item
 *   v = number of votes for this item
 *   m = minimum votes required (MINIMUM_VOTES)
 *   C = prior mean rating (PRIOR_MEAN)
 *
 * Examples:
 *   1 vote at 5.0: weighted = (1/6)*5.0 + (5/6)*3.0 = 0.83 + 2.5 = 3.33
 *   25 votes at 4.5: weighted = (25/30)*4.5 + (5/30)*3.0 = 3.75 + 0.5 = 4.25
 *
 *   So 25 votes at 4.5 (weighted 4.25) ranks higher than 1 vote at 5.0 (weighted 3.33)
 */
function calculateWeightedRating(averageRating, voteCount) {
  if (voteCount === 0) return PRIOR_MEAN;

  const weight = voteCount / (voteCount + MINIMUM_VOTES);
  return (weight * averageRating) + ((1 - weight) * PRIOR_MEAN);
}

const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
};

/**
 * Rate a wakeword (1-5 stars)
 * Uses visitor ID to prevent duplicate ratings
 */
exports.rate = async (event) => {
  try {
    const { id } = event.pathParameters;
    const body = JSON.parse(event.body);
    const { rating, visitorId } = body;

    // Validate rating
    if (!rating || rating < 1 || rating > 5 || !Number.isInteger(rating)) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Rating must be an integer from 1 to 5' }),
      };
    }

    if (!visitorId) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Visitor ID is required' }),
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

    // Check for existing rating from this visitor
    const existingRating = await docClient.send(new GetCommand({
      TableName: RATINGS_TABLE,
      Key: { wakewordId: id, visitorId },
    }));

    let ratingDelta = rating;
    let countDelta = 1;

    if (existingRating.Item) {
      // Update existing rating - calculate the difference
      ratingDelta = rating - existingRating.Item.rating;
      countDelta = 0; // Don't increment count for updates
    }

    // Save/update the rating
    await docClient.send(new PutCommand({
      TableName: RATINGS_TABLE,
      Item: {
        wakewordId: id,
        visitorId,
        rating,
        updatedAt: new Date().toISOString(),
      },
    }));

    // Update wakeword rating stats
    const current = wakeword.Item;
    const newRatingSum = (current.ratingSum || 0) + ratingDelta;
    const newRatingCount = (current.ratingCount || 0) + countDelta;
    const newAverageRating = newRatingCount > 0 ? newRatingSum / newRatingCount : 0;
    const newWeightedRating = calculateWeightedRating(newAverageRating, newRatingCount);

    await docClient.send(new UpdateCommand({
      TableName: WAKEWORDS_TABLE,
      Key: { id },
      UpdateExpression: 'SET ratingSum = :sum, ratingCount = :count, averageRating = :avg, weightedRating = :weighted, updatedAt = :now',
      ExpressionAttributeValues: {
        ':sum': newRatingSum,
        ':count': newRatingCount,
        ':avg': Math.round(newAverageRating * 100) / 100,
        ':weighted': Math.round(newWeightedRating * 100) / 100,
        ':now': new Date().toISOString(),
      },
    }));

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        message: existingRating.Item ? 'Rating updated' : 'Rating submitted',
        averageRating: Math.round(newAverageRating * 100) / 100,
        weightedRating: Math.round(newWeightedRating * 100) / 100,
        ratingCount: newRatingCount,
      }),
    };
  } catch (error) {
    console.error('Error rating wakeword:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to submit rating' }),
    };
  }
};
