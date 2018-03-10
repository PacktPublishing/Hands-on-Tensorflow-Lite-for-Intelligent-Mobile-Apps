/*
 * Packt Publishing
 * Hands-on Tensorflow Lite for Intelligent Mobile Apps
 * @author: Juan Miguel Valverde Martinez
 *
 * Section 3: Handwriting recognition
 * Video 3-5: Deployment in Android
 */

package com.example.lipman.handwrittingrecognition;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

//import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.Date;
import java.util.PriorityQueue;
import java.util.Queue;

public class MainActivity extends AppCompatActivity {

    /*********
     * Command use to freeze the graph:
     *
     * bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=../../3/tmp/graph.pb
     * --input_checkpoint=../../3/tmp/my-weights --input_binary=true
     * --output_graph=../../3/tmp/frozen.pb --output_node_names=prediction
     *
     *
     * Command use to convert the frozen model into tensorflow lite format.
     *
     * bazel-bin/tensorflow/contrib/lite/toco/toco --input_file=../../3/tmp/frozen.pb
     * --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE
     * --output_file=../../3/tmp/output3.lite --inference_type=FLOAT --input_type=FLOAT
     * --input_arrays=inputX --output_arrays=prediction --input_shapes=1,784
     *
     *********/

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

    static final int REQUEST_IMAGE_CAPTURE = 1;
    String mCurrentPhotoPath;
    Bitmap myImage;
    ImageView ivImage;
    int[][] myImageMatrix;
    static final int TARGET_DIM = 28;
    static final int NUM_CLASSES = 47;


    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnPicture = (Button) findViewById(R.id.btnPicture);
        Button btnResize = (Button) findViewById(R.id.btnResize);
        Button btnGrayscale = (Button) findViewById(R.id.btnGrayscale);
        Button btnThreshold = (Button) findViewById(R.id.btnThreshold);
        Button btnRecognition = (Button) findViewById(R.id.btnRecognition);
        Button btnAll = (Button) findViewById(R.id.btnAll);
        final TextView textResult = (TextView) findViewById(R.id.label);

        // Step 1: Take a picture
        btnPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Call a new activity
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                // Ensure that there's a camera activity to handle the intent

                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    // Create the File where the photo should go
                    File photoFile = null;
                    try {
                        photoFile = createImageFile();
                    } catch (IOException ex) {
                        // Error occurred while creating the File
                        Log.w("TAG","IOException!");
                    }
                    // Continue only if the File was successfully created
                    if (photoFile != null) {
                        Uri photoURI = FileProvider.getUriForFile(getBaseContext(),
                                "com.example.android.fileprovider2",
                                photoFile);
                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                    }
                }
            }
        });


        // Step 2: Resize
        btnResize.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                myImage = Bitmap.createScaledBitmap(myImage, TARGET_DIM, TARGET_DIM, true);
                ivImage.setImageBitmap(myImage);
            }
        });



        // Step 3: Convert to Grayscale
        btnGrayscale.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // Generate the 2D matrix containing grayscale values (int 0-255)
                myImageMatrix = getGrayscaleMatrix(myImage);
                // Display the grayscale matrix
                setGSMatrix(myImageMatrix);

            }
        });

        // Step 4: Threshold the grayscale image to have only black and white.
        //          0 --> Background. 1 --> Characters
        btnThreshold.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                myImageMatrix = thresholdMatrix(myImageMatrix,90);
                setGSMatrix(myImageMatrix);

            }
        });

        // Step 5: Recognition --> TF Lite.
        btnRecognition.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // Give it the appropriate size
                try {
                    // The interpreter will handle the graph
                    Interpreter tflite = new Interpreter(loadModelFile());
                    // Prepare the input
                    ByteBuffer inputData = matrix2bytebuffer();
                    // Declare where the output will be saved
                    float[][] output = new float[1][NUM_CLASSES];
                    // Run the graph
                    tflite.run(inputData, output);
                    // Close it to save resources
                    tflite.close();
                    textResult.setText(getLabels(output));

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        // All at once!
        btnAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                myImage = Bitmap.createScaledBitmap(myImage, TARGET_DIM, TARGET_DIM, true);
                ivImage.setImageBitmap(myImage);
                myImageMatrix = getGrayscaleMatrix(myImage);
                setGSMatrix(myImageMatrix);
                myImageMatrix = thresholdMatrix(myImageMatrix,90);
                setGSMatrix(myImageMatrix);

                try {
                    // The interpreter will handle the graph
                    Interpreter tflite = new Interpreter(loadModelFile());
                    // Prepare the input
                    ByteBuffer inputData = matrix2bytebuffer();
                    // Declare where the output will be saved
                    float[][] output = new float[1][NUM_CLASSES];
                    // Run the graph
                    tflite.run(inputData, output);
                    // Close it to save resources
                    tflite.close();
                    textResult.setText(getLabels(output));

                } catch (IOException e) {
                    e.printStackTrace();
                }

                  /*
                  // This works for tensorflow-mobile (we only need to freeze the graph)
                  TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(getAssets(),"frozen3.pb");
                  inferenceInterface.graphOperation("prediction");
                  inferenceInterface.feed("inputX",floatValues,1,784);
                  inferenceInterface.run(new String[] {"prediction"}, false);
                  inferenceInterface.fetch("prediction", outputs);
                  textResult.setText(getLabels(outputs));*/
            }
        });
    }

    // Modified from TF-Lite examples.
    private ByteBuffer matrix2bytebuffer() {
        // A float has 4 bytes.
        ByteBuffer data = ByteBuffer.allocateDirect(TARGET_DIM*TARGET_DIM*4);
        float[] floatValues = reshapeMatrix(myImageMatrix);

        data.order(ByteOrder.nativeOrder());
        data.rewind();
        for (int i=0;i<floatValues.length;i++)
        {
            data.putFloat(floatValues[i]);
        }

        return data;
    }

    /** Memory-map the model file in Assets. */
    // From TF-Lite examples.
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("output3.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    /**
     * This function returns the probabilities sorted by high chance.
     * @param probabilities
     * @return
     */
    public String getLabels(float[][] probabilities)
    {
        String[] labels = {"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","d","e","f","g","h","n","q","r","t"};
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

    /**
     * Reshape a 2D matrix into a vector (input of the model).
     * @param matrix
     * @return
     */
    public float[] reshapeMatrix(int[][] matrix)
    {
        float[] output = new float[matrix.length*matrix[0].length];
        for(int x = 0; x < matrix[0].length; ++x) { // width
            for (int y = 0; y < matrix.length; ++y) { // height
                output[y+x*matrix.length] = matrix[y][x];
            }
        }

        return output;
    }



    /**
     * This function thresholds the given matrix.
     * @param matrix
     * @param thr
     * @return
     */
    public int[][] thresholdMatrix(int[][] matrix, int thr)
    {
        int[][] result = new int[matrix[0].length][matrix.length];
        for(int x = 0; x < matrix[0].length; ++x) { // width
            for (int y = 0; y < matrix.length; ++y) { // height
                //Note: after this thresholding, we cannot see the image because "1" is a value close to black (0)
                result[y][x] = matrix[y][x]<thr ? 1 : 0;
            }
        }
        return result;
    }

    /**
     * This function sets the given 2D matrix (0-1) as the main image.
     * @param matrix
     */
    public void setGSMatrix(int[][] matrix)
    {
        Bitmap bmOut = Bitmap.createBitmap(matrix[0].length, matrix.length, Bitmap.Config.RGB_565);
        int c;
        for(int x = 0; x < matrix[0].length; ++x) { // width
            for (int y = 0; y < matrix.length; ++y) { // height
                c = matrix[y][x];
                bmOut.setPixel(x, y, Color.rgb(c,c,c));
            }
        }
        ivImage.setImageBitmap(bmOut);
    }

    /**
     * This function generates a 2D grayscale matrix given a ARGB bitmap.
     * @param src
     * @return
     */
    public int[][] getGrayscaleMatrix(Bitmap src)
    {
        int width, height, pixel;
        height = src.getHeight();
        width = src.getWidth();
        int R,G,B;
        int[][] matrix = new int[height][width];

        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                // Get one pixel color
                pixel = src.getPixel(x, y);
                // Retrieve color of all channels
                R = Color.red(pixel);
                G = Color.green(pixel);
                B = Color.blue(pixel);
                // Take conversion up to one single value
                matrix[y][x] = (int)(0.299 * R + 0.587 * G + 0.114 * B);

            }
        }
        return matrix;
    }



    /**
     * This function will be triggered when the picture is taken (and accepted by the user).
     * It will set the picture to the ImageView in the main activity and save the Bitmap.
     * @param requestCode
     * @param resultCode
     * @param data
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        ivImage = (ImageView) findViewById(R.id.picture);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // Show the thumbnail on ImageView
            Uri imageUri = Uri.parse(mCurrentPhotoPath);
            File file = new File(imageUri.getPath());
            try {
                InputStream ims = new FileInputStream(file);
                myImage = BitmapFactory.decodeStream(ims);
                ivImage.setImageBitmap(myImage);
            } catch (FileNotFoundException e) {
                return;
            }
        }
    }

    /**
     * This functions creates a file for the picture the camera will take.
     * @return created file.
     * @throws IOException
     */
    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = image.getAbsolutePath();
        return image;
    }
}
