/*

Based on TensferFlow exmaple
https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java\
Simplified to graph v3
==============================================================================*/


import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class LabelImageH3 {
    private static void printUsage(PrintStream s) {
        final String url =
                "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
        s.println(
                "Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)");
        s.println("to label JPEG images.");
        s.println("TensorFlow version: " + TensorFlow.version());
        s.println();
        s.println("Usage: label_image <model dir> <image file>");
        s.println();
        s.println("Where:");
        s.println("<model dir> is a directory containing the unzipped contents of the inception model");
        s.println("            (from " + url + ")");
        s.println("<image file> is the path to a JPEG image file");
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            printUsage(System.err);
            System.exit(1);
        }
        String modelDir = args[0];
        String imageFile = args[1];

        byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"));
        List<String> labels =
                readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"));
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));

        try (Tensor image = Tensor.create(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.println(
                    String.format(
                            "BEST MATCH: %s (%.2f%% likely)",
                            labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        }
    }


    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("final_result").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

}