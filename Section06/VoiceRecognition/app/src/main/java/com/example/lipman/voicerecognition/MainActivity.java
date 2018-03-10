/*
 * Packt Publishing
 * Hands-on Tensorflow Lite for Intelligent Mobile Apps
 * @author: Juan Miguel Valverde Martinez
 *
 * Section 6: Voice recognition
 * Video 6-4: Deployment in Android
 */

package com.example.lipman.voicerecognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.content.res.TypedArray;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Environment;
import android.renderscript.ScriptGroup;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.Date;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner;

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

    MediaRecorder recorder;
    String path;
    int NUM_CLASSES = 5;
    int TARGET_DIM = 10000;
    int times=1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button btnRecord = (Button) findViewById(R.id.btnRecord);
        Button btnStopRecord = (Button) findViewById(R.id.btnStopRecord);
        Button btnRecognition = (Button) findViewById(R.id.btnRecognition);
        final TextView tvStatus = (TextView) findViewById(R.id.status);
        final TextView textResult = (TextView) findViewById(R.id.label);

        // Step 3: Recognition
        btnRecognition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                // I would normally use the variable "path" to get the audio we've just recorded.
                float[] spectrogram = getSpectrogram(times+".wav");
                times++;
                if (times>5)
                    times=1;

                spectrogram = normalizeSpectrogram(spectrogram);

                // Give it the appropriate size
                try {
                    // The interpreter will handle the graph
                    Interpreter tflite = new Interpreter(loadModelFile());
                    // Prepare the input
                    ByteBuffer inputData = matrix2bytebuffer(spectrogram);
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

        // Step 2: Stop recording
        btnStopRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                recorder.stop();
                recorder.release();
                tvStatus.setText("Recorded");
                tvStatus.setTextColor(0xFF448c4e);
            }
        });

        // Step 1: Start recording
        btnRecord.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                recorder = new MediaRecorder();

                String status = Environment.getExternalStorageState();

                if(status.equals("mounted")) {
                    try {
                        path = createVoiceFile();
                        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                        recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                        recorder.setAudioEncoder(AudioFormat.ENCODING_PCM_16BIT); //AMR_NB
                        recorder.setOutputFile(path);
                        recorder.prepare();
                    } catch (IOException e) {
                        Log.w("Voice","errorr");
                        e.printStackTrace();
                    }

                    recorder.start();
                    tvStatus.setText("Recording...");
                    tvStatus.setTextColor(0xFFFF0000);
                }
            }
        });
    }

    /**
     * This function normalizes the spectrogram between 0 and 1, and pads or samples
     * the given spectrogram to fit the required length (10000).
     * @param spectrogram
     * @return
     */
    private float[] normalizeSpectrogram(float[] spectrogram)
    {
        // Finding max and min to normalize
        float cmax=-999999, cmin=999999;
        float[] finalSpectrogram = new float[TARGET_DIM];
        for (int i=0;i<spectrogram.length;i++)
        {
            if (spectrogram[i]>cmax)
                cmax=spectrogram[i];
            if (spectrogram[i]<cmin)
                cmin=spectrogram[i];
        }
        int i,index;
        if (spectrogram.length<TARGET_DIM) {
            int offset = (TARGET_DIM-spectrogram.length)/2;
            for (i=0;i<offset;i++)
            {
                finalSpectrogram[i] = 0;
            }
            for (;i<spectrogram.length+offset;i++)
            {
                finalSpectrogram[i] = spectrogram[i-offset];
            }
            for (;i<TARGET_DIM;i++)
            {
                finalSpectrogram[i] = 0;
            }

        } else {
            for (i=0;i<TARGET_DIM;i++)
            {
                // Taking samples equally spaced
                index = ( i*spectrogram.length/TARGET_DIM) + (spectrogram.length/(TARGET_DIM*2));
                finalSpectrogram[i] = spectrogram[index];
            }
        }

        for (i=0;i<TARGET_DIM;i++)
        {
            finalSpectrogram[i] = (cmax-finalSpectrogram[i])/(cmax-cmin);
        }
        return finalSpectrogram;
    }

    // Modified from TF-Lite examples.
    private ByteBuffer matrix2bytebuffer(float[] floatValues) {
        // A float has 4 bytes.
        ByteBuffer data = ByteBuffer.allocateDirect(TARGET_DIM*4);

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
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("output6.tflite");
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
        String[] labels = {"okay","hi","go","left","right"};
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

    private float[] getSpectrogram(String pathFile)
    {
        AssetManager mngr = getAssets();

        byte[] musicBytes = new byte[512];
        byte[] fullMusicBytes = new byte[512];
        byte[] tempMusicBytes;
        int c = 0;

        try {
            InputStream is = mngr.open(pathFile);
            while (is.available() > 0) {
                c++;
                is.read(musicBytes);
                if (fullMusicBytes != null) {
                    tempMusicBytes = new byte[fullMusicBytes.length + musicBytes.length];
                    System.arraycopy(fullMusicBytes, 0, tempMusicBytes, 0, fullMusicBytes.length);
                    System.arraycopy(musicBytes, 0, tempMusicBytes, fullMusicBytes.length, musicBytes.length);
                    fullMusicBytes = tempMusicBytes;
                } else {
                    fullMusicBytes = musicBytes;
                }

            }
            is.close();
        } catch (Exception e) {

        }

        int totalSamples = ByteBuffer.wrap(new byte[]{fullMusicBytes[552 + 3], fullMusicBytes[552+ 2], fullMusicBytes[552 + 1], fullMusicBytes[552]}).getInt()/2;

        int c1 = 556, c2 = 0;
        byte[] b;
        float[] spectrogram = new float[totalSamples];

        while (c1 < totalSamples*2+556) {
            b = new byte[]{0, 0, fullMusicBytes[c1 + 1], fullMusicBytes[c1]};
            spectrogram[c2] = ByteBuffer.wrap(b).getInt();
            // Convert into two's complement: https://en.wikipedia.org/wiki/Two%27s_complement
            if ((fullMusicBytes[c1 + 1] & 0x80) == 128)
                spectrogram[c2] -= 65536;
            c1 += 2;
            c2++;
        }

        return spectrogram;
    }

    /**
     * This functions creates a file for the picture the camera will take.
     * @return created file.
     * @throws IOException
     */
    private String createVoiceFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String voiceFileName = "VOICE_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PODCASTS);
        File voice = File.createTempFile(
                voiceFileName,  /* prefix */
                ".wav",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        return voice.getAbsolutePath();
    }
}
