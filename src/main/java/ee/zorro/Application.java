package ee.zorro;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.List;

public class Application {
    public static void main(String[] args) throws Exception {
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
