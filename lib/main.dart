import 'package:flutter/material.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:csv/csv.dart';
import 'package:flutter/services.dart' show rootBundle;


void main() async{

  final samples = await fromCsv('assets/datasets/diabetes.csv', headerExists: true);
  final targetColumnName = 'class variable (0 or 1)';
  final splits = splitData(samples, [0.7]);
  final validationData = splits[0];
  final testData = splits[1];
  final validator = CrossValidator.kFold(validationData, [targetColumnName], numberOfFolds: 5);
  final createClassifier = (DataFrame samples, _) =>
      LogisticRegressor(
        samples,
        'targetColumnName',
        optimizerType: LinearOptimizerType.gradient,
        iterationsLimit: 90,
        initialLearningRate: 0.05,
        learningRateType: LearningRateType.decreasingAdaptive,
        batchSize: samples.rows.length,
        probabilityThreshold: 0.7,
        collectLearningData: true,
      );
  final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
  final accuracy = scores.mean();

  print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');

  final testSplits = splitData(testData, [0.8]);
  final classifier = createClassifier(testSplits[0], [targetColumnName]);
  final finalScore = classifier.assess(testSplits[1], [targetColumnName], MetricType.accuracy);

  print(finalScore.toStringAsFixed(2));

  await classifier.saveAsJson('diabetes_classifier.json');

}
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: FloatingActionButton(onPressed: null),
);
}
}
