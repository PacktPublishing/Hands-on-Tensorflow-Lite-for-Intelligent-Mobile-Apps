package com.example.lipman.gesturerecognition;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
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
import java.util.HashMap;
import java.util.Map;
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

    static final int REQUEST_IMAGE_CAPTURE = 1;
    String mCurrentPhotoPath;
    Bitmap myImage,myBackground, myImageCropped;
    ImageView ivImage;
    int[][] subtraction;
    static final int TARGET_DIM = 60;
    // After subtracting, if the difference is larger than this, it will be part of the foreground.
    static final int THR_SUB = 50;
    static final int NUM_CLASSES = 3;

    boolean firstTime = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ivImage = (ImageView) findViewById(R.id.picture);
        Button btnPicture = (Button) findViewById(R.id.btnPicture);
        Button btnSubtract = (Button) findViewById(R.id.btnSubtract);
        Button btnCropScale = (Button) findViewById(R.id.btnCropScale);
        Button btnRecognition = (Button) findViewById(R.id.btnRecognition);
        final TextView textResult = (TextView) findViewById(R.id.label);


        btnRecognition.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                try {
                    Map<Integer,Object> newOutput = new HashMap<Integer,Object>();

                    // The interpreter will handle the graph
                    Interpreter tflite = new Interpreter(loadModelFile());
                    // Prepare the input
                    ByteBuffer[] inputData = matrix2bytebuffer();
                    // Declare where the output will be saved
                    float[][] output = new float[1][NUM_CLASSES];
                    newOutput.put(0,output);
                    // Run the graph
                    tflite.runForMultipleInputsOutputs(inputData,newOutput);

                    //tflite.run(inputData, output);
                    // Close it to save resources
                    tflite.close();

                    textResult.setText(getLabels(output));

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        // Step 3: Crop and scale
        btnCropScale.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                //Crop the image.
                int[] offsets = getOffsetfromSubtraction();
                subtraction = cropSubtraction(offsets);
                // Adjust its size
                adjustSize();

                myImageCropped = matrix2image(subtraction);
                myImageCropped = Bitmap.createScaledBitmap(myImageCropped,
                        TARGET_DIM, TARGET_DIM, true);

                ivImage.setImageBitmap(myImageCropped);
            }
        });

        // Step 2: Background subtraction
        btnSubtract.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // Generate the 2D matrix containing grayscale values (int 0-255)
                // myImage-myBackground --> new pixels will be part of the foreground.
                subtraction = new int[myImage.getHeight()][myImage.getWidth()];
                subtractBackground();

                ivImage.setImageBitmap(matrix2image(subtraction));

            }
        });

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
                                "com.example.android.fileprovider4",
                                photoFile);
                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                    }
                }
            }
        });

    }

    /**
     * This function returns the probabilities sorted by high chance.
     * @param probabilities
     * @return
     */
    public String getLabels(float[][] probabilities)
    {
        String[] labels = {"0","1","2"};
        Result res;
        String results = "";
        Log.w("lipman",""+probabilities[0][0]);
        Log.w("lipman",""+probabilities[0][1]);
        Log.w("lipman",""+probabilities[0][2]);

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
     * This function prepares the input data for the model (3 inputs).
     * @return
     */
    private ByteBuffer[] matrix2bytebuffer() {
        // Input 1
        ByteBuffer[] data = new ByteBuffer[3];
        ByteBuffer data1 = ByteBuffer.allocateDirect(TARGET_DIM*TARGET_DIM*4);
        float[] floatValues = prepareInput1();

        data1.order(ByteOrder.nativeOrder());
        data1.rewind();
        for (int i=0;i<floatValues.length;i++)
        {
            data1.putFloat(floatValues[i]);
        }
        data[0] = data1;

        // Input 2
        ByteBuffer data2 = ByteBuffer.allocateDirect(TARGET_DIM*4);
        float[] floatValues2 = prepareInput2();

        data2.order(ByteOrder.nativeOrder());
        data2.rewind();
        for (int i=0;i<floatValues2.length;i++)
        {
            data2.putFloat(floatValues2[i]);
        }
        data[1] = data2;

        // Input 3
        ByteBuffer data3 = ByteBuffer.allocateDirect(TARGET_DIM*4);
        float[] floatValues3 = prepareInput3();

        data3.order(ByteOrder.nativeOrder());
        data3.rewind();
        for (int i=0;i<floatValues3.length;i++)
        {
            data3.putFloat(floatValues3[i]);
        }
        data[2] = data3;


        return data;
    }

    /**
     * Preparing the second input tensor.
     * @return
     */
    public float[] prepareInput2() {
        float[] output = new float[TARGET_DIM];
        int pixel,tmpVal;
        for (int x=0;x<TARGET_DIM;x++)
        {
            for (int y=0;y<TARGET_DIM;y++)
            {
                pixel = myImageCropped.getPixel(x,y);
                tmpVal = Color.red(pixel)>0 ? 1 : 0;
                output[x] += tmpVal;
            }
        }
        return output;
    }

    /**
     * Preparing the third input tensor.
     * @return
     */
    public float[] prepareInput3() {
        float[] output = new float[TARGET_DIM];
        int pixel,tmpVal;
        for (int x=0;x<TARGET_DIM;x++)
        {
            for (int y=0;y<TARGET_DIM;y++)
            {
                pixel = myImageCropped.getPixel(y,x);
                tmpVal = Color.red(pixel)>0 ? 1 : 0;
                output[x] += tmpVal;
            }
        }
        return output;
    }

    /**
     * Preparing the first input tensor.
     * @return
     */
    public float[] prepareInput1() {
        float[] output = new float[TARGET_DIM*TARGET_DIM];
        int pixel;
        for (int x=0;x<TARGET_DIM;x++)
        {
            for (int y=0;y<TARGET_DIM;y++)
            {
                pixel = myImageCropped.getPixel(x,y);
                output[x*TARGET_DIM+y] = Color.red(pixel)>0 ? 1 : 0;
            }
        }
        return output;
    }

    /** Memory-map the model file in Assets. */
    // From TF-Lite examples.
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("output5.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    /**
     * This function will create a squared matrix to contain the cropped subtraction.
     * If we don't do this, when adjusting our cropped image to fit the model, the axis may be
     * stretched.
     */
    public void adjustSize()
    {
        int maxValue = subtraction.length > subtraction[0].length ? subtraction.length : subtraction[0].length;
        int[][] newSubtraction = new int[maxValue][maxValue];
        int offsetX = (maxValue-subtraction[0].length)/2;
        int offsetY = (maxValue-subtraction.length)/2;
        for (int x=0;x<subtraction.length;x++)
        {
            for (int y=0;y<subtraction[0].length;y++)
            {
                newSubtraction[x+offsetY][y+offsetX] = subtraction[x][y];
            }
        }
        subtraction = newSubtraction;
    }

    /**
     * This function will crop "subtraction".
     * @param offsets to know where we can crop.
     * @return new cropped subtraction.
     */
    public int[][] cropSubtraction(int[] offsets){
        int[][] result = new int[offsets[1]-offsets[0]][offsets[3]-offsets[2]];
        for (int x=0;x<result.length;x++)
        {
            for (int y=0;y<result[0].length;y++)
            {
                result[x][y] = subtraction[offsets[0]+x][offsets[2]+y];
            }
        }
        return result;
    }

    /**
     * This function will find the coordinates of "subtraction" where we can crop. For this it will
     * iterate top-down, down-top, left-right, right-left to find the first non-background pixel.
     * This way we can crop the interesting part (hand) and eliminate remaining pixels.
     * @return
     */
    public int[] getOffsetfromSubtraction(){
        int[] offsets = new int[4];
        // Background is 0
        int tmpsum=0,iter=0;
        // Find the offset top-down
        while (tmpsum==0)
        {
            for (int i=0;i<myBackground.getWidth();i++)
            {
                tmpsum+=subtraction[iter][i];
            }
            iter++;
        }
        offsets[0] = iter;

        // Find the offset down-top
        iter=myBackground.getHeight()-1;tmpsum=0;
        while (tmpsum==0)
        {
            for (int i=0;i<myBackground.getWidth();i++)
            {
                tmpsum+=subtraction[iter][i];
            }
            iter--;
        }
        offsets[1] = iter;

        // Find the offset left-right
        iter=0;tmpsum=0;
        while (tmpsum==0)
        {
            for (int i=0;i<myBackground.getHeight();i++)
            {
                tmpsum+=subtraction[i][iter];
            }
            iter++;
        }
        offsets[2] = iter;

        // Find the offset right-left
        iter=myBackground.getWidth()-1;tmpsum=0;
        while (tmpsum==0)
        {
            for (int i=myBackground.getHeight()-1;i>=0;i--)
            {
                tmpsum+=subtraction[i][iter];
            }
            iter--;
        }
        offsets[3] = iter;

        return offsets;
    }

    /**
     * This function will subtract the background (myBackground) from the current image (myImage).
     */
    public void subtractBackground()
    {
        int pixelBackground,pixelForeground,diff1,diff2,diff3;
        for (int x=0; x < myBackground.getHeight(); x++)
        {
            for (int y=0; y < myBackground.getWidth(); y++)
            {
                pixelBackground = myBackground.getPixel(y, x);
                pixelForeground = myImage.getPixel(y, x);
                diff1 = Color.red(pixelBackground)-Color.red(pixelForeground);
                diff2 = Color.green(pixelBackground)-Color.green(pixelForeground);
                diff3 = Color.blue(pixelBackground)-Color.blue(pixelForeground);
                subtraction[x][y] = Math.abs(diff1)+Math.abs(diff2)+Math.abs(diff3) > THR_SUB ? 255 : 0;
            }
        }
    }

    /**
     * This function sets the given 2D matrix (0-255) as the main image.
     * @param matrix
     */
    public Bitmap matrix2image(int[][] matrix)
    {
        Bitmap bmOut = Bitmap.createBitmap(matrix[0].length, matrix.length, Bitmap.Config.RGB_565);
        int c;
        for(int x = 0; x < matrix[0].length; ++x) { // width
            for (int y = 0; y < matrix.length; ++y) { // height
                c = matrix[y][x];
                bmOut.setPixel(x, y, Color.rgb(c,c,c));
            }
        }
        return bmOut;
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


        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // Show the thumbnail on ImageView
            Uri imageUri = Uri.parse(mCurrentPhotoPath);
            File file = new File(imageUri.getPath());
            try {
                InputStream ims = new FileInputStream(file);
                if (firstTime) {
                    myBackground = BitmapFactory.decodeStream(ims);
                    ivImage.setImageBitmap(myBackground);
                    firstTime = false;
                } else {
                    myImage = BitmapFactory.decodeStream(ims);
                    ivImage.setImageBitmap(myImage);
                }
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
