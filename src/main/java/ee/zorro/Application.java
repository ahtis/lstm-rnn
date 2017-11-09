package ee.zorro;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.IOException;
import java.util.List;

import static org.nd4j.linalg.lossfunctions.LossFunctions.*;

public class Application {
    public static void main(String[] args) throws Exception {
//        int numRows = 28;
//        int numColumns = 28;
//        int outputNum = 10;
//        int batchSize = 128;
//        int rngSeed = 123;
//        int numEpochs = 15;

//        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
//        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false,  rngSeed);
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngSeed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .learningRate(0.006)
//                .updater(new Nesterovs(0.9))
//                .regularization(true)
//                .l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder()
//                        .nIn(numRows * numColumns)
//                        .nOut(1000)
//                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.XAVIER)
//                        .build()
//                )
//                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nIn(1000)
//                        .nOut(outputNum)
//                        .activation(Activation.SOFTMAX)
//                        .weightInit(WeightInit.XAVIER)
//                        .build()
//                )
//                .pretrain(false)
//                .backprop(true)
//                .build();
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        model.setListeners(new ScoreIterationListener(1));
//
//        for (int i = 0; i < numEpochs; i++) {
//            model.fit(mnistTrain);
//
//            Evaluation eval = model.evaluate(mnistTest);
//            System.out.println(eval.stats());
//            mnistTest.reset();
//        }

        int numLinesToSkip = 0;
        char delimiter = ',';
        char quote = '"';

        Schema schema = new Schema.Builder()
            .addColumnString("recordDate")
            .addColumnInteger("passengers")
            .build();
        System.out.println(schema);

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
            .stringToTimeTransform("recordDate", "YYYY-MM", DateTimeZone.UTC)
            .transform(
                new DeriveColumnsFromTimeTransform.Builder("recordDate")
                    .addIntegerDerivedColumn("year", DateTimeFieldType.year())
                    .addIntegerDerivedColumn("month", DateTimeFieldType.monthOfYear())
                    .build()
            )
            .removeColumns("recordDate")
            .build();

        Schema outputSchema = transformProcess.getFinalSchema();
        System.out.println(outputSchema);

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("International Airline Passengers Prediction");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter, quote);
        String path = new ClassPathResource("InternationalAirlinePassengers/international-airline-passengers.csv").getFile().getAbsolutePath();

        JavaRDD<String> stringData = sparkContext.textFile(path);
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(recordReader));
        JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, transformProcess);
        JavaRDD<String> processedAsString = processedData.map(new WritablesToStringFunction(String.valueOf(delimiter)));

        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = stringData.collect();

        System.out.println("\n\n----- Original Data ------");
        for (String s: inputDataCollected) {
            System.out.println(s);
        }

        System.out.println("\n\n----- Processed Data ------");
        for (String s: processedCollected) {
            System.out.println(s);
        }
    }
}
