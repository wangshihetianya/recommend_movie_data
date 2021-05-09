package com.recsys.offline.spark

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object CollaborativeFiltering {

  def main(args: Array[String]): Unit = {
    var conf = new SparkConf()

    var ratingResourcesPath :String = null

    if (args.length < 1) {
      conf.setMaster("local")
        .setAppName("collaborativeFiltering")
        .set("spark.submit.deployMode", "client")
      ratingResourcesPath = this.getClass.getResource("/sampledata/ratings.csv").getPath
    } else {
      conf.setMaster("spark://master:7077")
        .setAppName("collaborativeFiltering")
      ratingResourcesPath = args(0)
    }

    val spark = SparkSession.builder.config(conf).getOrCreate()
//    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val toInt = udf[Int, String]( _.toInt)
    val toFloat = udf[Double, String]( _.toFloat)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath)
      .withColumn("userIdInt", toInt(col("userId")))
      .withColumn("movieIdInt", toInt(col("movieId")))
      .withColumn("ratingFloat", toFloat(col("rating")))

    val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userIdInt")
      .setItemCol("movieIdInt")
      .setRatingCol("ratingFloat")

    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    model.itemFactors.show(10, truncate = false)
    model.userFactors.show(10, truncate = false)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("ratingFloat")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

    // Generate top 10 movie recommendations for a specified set of users
    val users = ratingSamples.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)
    // Generate top 10 user recommendations for a specified set of movies
    val movies = ratingSamples.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    // $example off$
    userRecs.show(false)
    movieRecs.show(false)
    userSubsetRecs.show(false)
    movieSubSetRecs.show(false)

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.regParam, Array(0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(als)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)  // Use 3+ in practice
    val cvModel = cv.fit(test)
    val avgMetrics = cvModel.avgMetrics

    spark.stop()
  }
}