/*
 * Packt Publishing
 * Hands-on Tensorflow Lite for Intelligent Mobile Apps
 * @author: Juan Miguel Valverde Martinez
 *
 * Section 4: Pattern recognition
 * Video 4-5: Deployment in Android
 */

package com.example.lipman.patternrecognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;


public class MainActivity extends AppCompatActivity {

    public class Result {
        String label;
        float probability;
        public Result(String label, float probability) {
            this.label = label;
            this.probability = probability;
        }
        public float getProbability() { return this.probability; }
    }

    public static Comparator<Result> idComparator = new Comparator<Result>(){
        @Override
        public int compare(Result c1, Result c2) {
            return (int) (c1.getProbability()*10000 - c2.getProbability()*10000);
        }
    };

    private int currentPic = 0;
    Bitmap myImage, myImageCropped;
    ImageView ivImage;
    static final int TARGET_DIM = 40;
    static final int TARGET_LENGTH = 100;
    static final int NUM_CLASSES = 13;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnPicture = (Button) findViewById(R.id.btnPicture);
        Button btnResize = (Button) findViewById(R.id.btnResize);
        Button btnRecognition = (Button) findViewById(R.id.btnRecognition);
        ivImage = (ImageView) findViewById(R.id.picture);

        // Step 3: Convert to Grayscale
        btnRecognition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Perform the recognition

                try {
                    // The interpreter will handle the graph
                    Interpreter tflite = new Interpreter(loadModelFile());
                    // Prepare the input. Resize it to the same dimensions that the network expects.
                    ByteBuffer inputData = bitmap2bytebuffer(myImageCropped);
                    // Declare where the output will be saved
                    float[][] output = new float[1][NUM_CLASSES];
                    // Run the graph
                    tflite.run(inputData, output);
                    // Close it to save resources
                    tflite.close();

                    showLabels(output);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        // Step 2: Resize
        btnResize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Crop an square portion of the picture from the center of size TARGET_LENGTH
                int offsetX = (myImage.getWidth()-TARGET_LENGTH)/2;
                int offsetY = (myImage.getHeight()-TARGET_LENGTH)/2;
                myImageCropped = Bitmap.createBitmap(myImage,offsetX,offsetY,
                        TARGET_LENGTH,TARGET_LENGTH);

                // Resize to ensure it will have the desired dimensions
                myImageCropped = Bitmap.createScaledBitmap(myImageCropped,
                        TARGET_DIM, TARGET_DIM, true);

                ivImage.setImageBitmap(myImageCropped);

            }
        });

        // Step 1: Get a picture from all the pictures I have
        btnPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Get picture
                // Establish a counter to keep track of the current picture
                if (currentPic<3)
                    currentPic++;
                else
                    currentPic=1;


                Resources resources = getBaseContext().getResources();
                int resourceId = resources.getIdentifier("test"+currentPic, "drawable",getBaseContext().getPackageName());
                ivImage.setImageResource(resourceId);
                // To not resize and keep my dimensions.
                BitmapFactory.Options o = new BitmapFactory.Options();
                o.inScaled = false;
                myImage = BitmapFactory.decodeResource(resources,resourceId, o);
            }
        });
    }

    /**
     * This function will get the probabilities in the given float, sort them and show them in
     * the Textview.
     * @param probabilities
     */
    public void showLabels(float[][] probabilities)
    {
        String[] labels = {"0","1","2","3","4","5","6","7","8","9","10","11","12"};
        Result res;
        String results = "";
        TextView tx1 = (TextView) findViewById(R.id.label1);

        Queue<Result> pq = new PriorityQueue<>(probabilities[0].length, idComparator);
        for (int i=0;i<probabilities[0].length;i++) {
            pq.add(new Result(labels[i],probabilities[0][i]));
        }

        for (int i=0;i<probabilities[0].length;i++) {
            res = pq.poll();
            results = res.label+","+results;
        }
        tx1.setText(results);
    }

    // Modified from TF-Lite examples.
    private ByteBuffer bitmap2bytebuffer(Bitmap bitmap) {
        // A float has 4 bytes. 3 channels.
        ByteBuffer data = ByteBuffer.allocateDirect(TARGET_DIM*TARGET_DIM*3*4);

        int[] intValues = new int[TARGET_DIM * TARGET_DIM];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        data.order(ByteOrder.nativeOrder());
        data.rewind();
        int R,G,B;
        for (int i=0;i<TARGET_DIM*TARGET_DIM;i++)
        {
            int val = intValues[i];
            R = Color.red(val);
            G = Color.green(val);
            B = Color.blue(val);

            data.putFloat(R);
            data.putFloat(G);
            data.putFloat(B);
        }

        return data;
    }

    /** Memory-map the model file in Assets. */
    // From TF-Lite examples.
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("output4.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

}
