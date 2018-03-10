package com.example.lipman.patternrecognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;


public class oldMainActivity extends AppCompatActivity {

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

    private int currentPic = -1;
    Bitmap myImage;
    ImageView ivImage;
    int[][][] myImageMatrix;
    static final int TARGET_DIM = 40;
    static final int TARGET_LENGTH = 150;
    static final int NUM_CLASSES = 13;
    ArrayList<Bitmap> crops;
    int crop1[][][];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnPicture = (Button) findViewById(R.id.btnPicture);
        Button btnResize = (Button) findViewById(R.id.btnResize);
        Button btnRecognition = (Button) findViewById(R.id.btnRecognition);
        ivImage = (ImageView) findViewById(R.id.picture);

        crop1 = new int[TARGET_DIM][TARGET_DIM][3];

        // Step 3: Convert to Grayscale
        btnRecognition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Perform the recognition
                // from myImageMatrix, get the 4 40x40x3 pieces.
                // Give it the appropriate size

                try {

                    for (int a = 0; a < 4; a++) {
                        // The interpreter will handle the graph
                        Interpreter tflite = new Interpreter(loadModelFile());
                        // Prepare the input
                        ByteBuffer inputData = matrix2bytebuffer(Bitmap.createBitmap(crops.get(a), 0, 0,
                                TARGET_DIM, TARGET_DIM));
                        // Declare where the output will be saved
                        float[][] output = new float[1][NUM_CLASSES];
                        // Run the graph
                        tflite.run(inputData, output);
                        // Close it to save resources
                        tflite.close();
                        //textResult.setText(getLabels(output));

                        //Log.w("lipman", getLabels(output));
                        Log.w("lipman",""+output[0][0]);
                        for (int r=0;r<13;r++)
                        {
                            Log.w("lipman",r+": "+output[0][r]);
                        }
                    }

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        // Step 2: Resize
        btnResize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Take the center 200x200, divide it by 4 100x100, and resize each of them
                // to 40x40 to put it in the network.
                // Crop myImage
                int offsetX = (myImage.getWidth()-2*TARGET_LENGTH)/2;
                int offsetY = (myImage.getHeight()-2*TARGET_LENGTH)/2;
                Bitmap myImageCropped = Bitmap.createBitmap(myImage,offsetX,offsetY,
                        myImage.getWidth()-offsetX*2,myImage.getHeight()-offsetY*2);

                // Resize to ensure it will have the desired dimensions
                myImageCropped = Bitmap.createScaledBitmap(myImageCropped,
                        TARGET_LENGTH*2, TARGET_LENGTH*2, true);

                // Set image bitmap
                ivImage.setImageBitmap(myImageCropped);
                crops = new ArrayList<Bitmap>();
                crops.add(Bitmap.createBitmap(myImage,0,0,
                        TARGET_LENGTH,TARGET_LENGTH));
                crops.add(Bitmap.createBitmap(myImage,0,TARGET_LENGTH,
                        TARGET_LENGTH,TARGET_LENGTH));
                crops.add(Bitmap.createBitmap(myImage,TARGET_LENGTH,0,
                        TARGET_LENGTH,TARGET_LENGTH));
                crops.add(Bitmap.createBitmap(myImage,TARGET_LENGTH,TARGET_LENGTH,
                        TARGET_LENGTH,TARGET_LENGTH));


                // Get 3D matrix
                //myImageMatrix = bitmap2matrix(myImageCropped);
                crop1 = bitmap2matrix(crops.get(0));

            }
        });

        // Step 1: Get a picture from all the pictures I have
        btnPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Get picture
                // Establish a counter to keep track of the current picture
                if (currentPic<3)
                    currentPic++;
                else
                    currentPic=0;


                Resources resources = getBaseContext().getResources();
                int resourceId = resources.getIdentifier("img"+currentPic, "drawable",getBaseContext().getPackageName());
                ivImage.setImageResource(resourceId);
                myImage = BitmapFactory.decodeResource(resources,resourceId);
            }
        });
    }

    public String getLabels(float[][] probabilities)
    {
        String[] labels = {"0","1","2","3","4","5","6","7","8","9","10","11","12"};
        Result res;
        String results = "";

        Queue<Result> pq = new PriorityQueue<>(probabilities[0].length, idComparator);
        for (int i=0;i<probabilities[0].length;i++) {
            pq.add(new Result(labels[i],probabilities[0][i]));
        }

        for (int i=0;i<probabilities[0].length;i++) {
            res = pq.poll();
            results = res.label+","+results;
        }

        return results;
    }

    // Modified from TF-Lite examples.
    private ByteBuffer matrix2bytebuffer(Bitmap bitmap) {
        // A float has 4 bytes.
        // A float has 4 bytes.
        ByteBuffer data = ByteBuffer.allocateDirect(TARGET_DIM*TARGET_DIM*4*3);

        int[] intValues = new int[TARGET_DIM * TARGET_DIM];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        data.order(ByteOrder.nativeOrder());
        data.rewind();
        int pixel = 0;
        for (int i=0;i<bitmap.getWidth();i++)
        {
            for (int j=0;j<bitmap.getHeight();j++)
            {
                // val is a pixel from the bitmap (32 bits)
                // 1 byte per channel: alpha, RGB
                final int val = intValues[pixel++];
                // We take the red byte, move it to the left, and mask the rest out
                data.put((byte) ((val >> 16) & 0xFF));
                // Same with the green byte, we isolate it
                data.put((byte) ((val >> 8) & 0xFF));
                // .. and the same with the blue byte.
                data.put((byte) (val & 0xFF));
            }
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

    public int[][][] bitmap2matrix(Bitmap src)
    {
        int width, height, pixel;
        height = src.getHeight();
        width = src.getWidth();
        int R,G,B;
        int[][][] matrix = new int[height][width][3];

        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                // Get one pixel color
                pixel = src.getPixel(x, y);
                // Retrieve color of all channels
                matrix[y][x][0] = Color.red(pixel);
                matrix[y][x][1] = Color.green(pixel);
                matrix[y][x][2] = Color.blue(pixel);

            }
        }
        return matrix;
    }


}
