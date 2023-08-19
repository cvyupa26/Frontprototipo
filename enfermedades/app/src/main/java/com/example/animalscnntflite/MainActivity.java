package com.example.animalscnntflite;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
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
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    ImageView imgInput;
    TextView txtPrediccion;
    Button btnElegir, btnPredecir;

    Interpreter tflite;
    Interpreter.Options options = new Interpreter.Options();
    ByteBuffer imgData = null;
    /*
    ByteBuffer: Es solo un contenedor o tanque de almacenamiento para leer o escribir datos en.
    Se le asignan datos utilizando la API allocateDirect()
    */
    private static int IMG_SIZE = 32;
    private static int NUM_CLASSES = 3;
    private static int BATCH_SIZE = 1;
    private static int PIXEL_SIZE = 3; //3 canales

    int[] imgPixels = new int[IMG_SIZE*IMG_SIZE]; //(32*32) 1024px
    float[][] result = new float[1][NUM_CLASSES];

    //constantes para permisos de la camara
    private static final int IMAGE_PICK_CODE = 1000;
    private static final int PERMISSION_CODE = 1001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgInput = findViewById(R.id.imgInput);
        txtPrediccion = findViewById(R.id.txtPrediccion);
        btnElegir = findViewById(R.id.btnElegir);
        btnPredecir = findViewById(R.id.btnPredecir);
        //Inicializar interpreter tensorflow-lite
        try {
            tflite = new Interpreter(loadModelFile(), options);
        }catch (Exception e){
            e.printStackTrace();
        }
        //https://bit.ly/2YgAqeG (explica los parámetros de ByteBuffer.allocateDirect(x,x,x,x) )
        imgData = ByteBuffer.allocateDirect(4 * BATCH_SIZE * IMG_SIZE * IMG_SIZE * PIXEL_SIZE); //Crea un ByteBuffer con una capacidad (4*1*32*32*3) = 12288 Bytes
        /*
        * ByteBuffer representa la imagen como una matriz 1D con 3 bytes por canal (rojo, verde y azul).
        * Llamamos order (ByteOrder.nativeOrder ()) para asegurarnos de que los bits se almacenan en el orden nativo del dispositivo.
        * */
        //Para imágenes del dataset MNIST:  imgData = ByteBuffer.allocateDirect(4*28*28);
        imgData.order(ByteOrder.nativeOrder());

        //Elegir imagen
        btnElegir.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(Build.VERSION.SDK_INT == Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) ==
                            PackageManager.PERMISSION_DENIED){
                        String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                        requestPermissions(permissions, PERMISSION_CODE);
                    }else{
                        chooseImageFromGallery();
                    }
                }else{
                    chooseImageFromGallery();
                }
            }
        });
        //boton predecir
        btnPredecir.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imgInput.invalidate();
                BitmapDrawable drawable = (BitmapDrawable) imgInput.getDrawable();
                Bitmap bitmap = drawable.getBitmap();
                Bitmap bitmap_resize = getResizedBitmap(bitmap, IMG_SIZE, IMG_SIZE); //bitmap.. ancho, altura
                convertBitmapToByteBuffer(bitmap_resize);

                tflite.run(imgData, result);
                txtPrediccion.setText("");
                String labels[] = {"GATO","PERRO","PANDA"};
                //txtPrediccion.setText("result= "+ Arrays.toString(result[0]));
                //txtPrediccion.setText("result= " + argmax(result[0])+" normal=> "+Arrays.toString(result[0]));

                txtPrediccion.setText(labels[argmax(result[0])]+"\nProbs:"+Arrays.toString(result[0]));
            }
        });
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("animals.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void chooseImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE) {
            assert data != null;
            imgInput.setImageURI(data.getData());
        }
    }

    // runtime
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_CODE) {
            chooseImageFromGallery();
        }
    }

    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight){
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float)newWidth)/width;
        float scaleHeight = ((float)newHeight)/height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        return Bitmap.createBitmap(bm, 0,0,width, height, matrix, false);
    }
    /*
    * El código de convertBitmapToByteBuffer procesa previamente las imágenes de mapa de bits entrantes
    * de la cámara a este ByteBuffer. Llama al método addPixelValue/convertPixel para agregar cada conjunto de valores
    * de píxeles al ByteBuffer secuencialmente
    * */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        //rebobinar el búfer
        imgData.rewind();
        bitmap.getPixels(imgPixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < IMG_SIZE; ++i) {
            for (int j = 0; j < IMG_SIZE; ++j) {
                int value = imgPixels[pixel++];
                //imgData.putFloat(convertPixel(value));
                addPixelValue(value);
            }
        }
    }


    //Original: usado en imágenes de dataset MNIST (blanco y negro)
      /*
        private static float convertPixel(int color) {
            return (255 - (((color >> 16) & 0xFF) * 0.299f
                    + ((color >> 8) & 0xFF) * 0.587f
                    + (color & 0xFF) * 0.114f)) / 255.0f;
        }
        */
   /*
   * Para ClassifierFloatMobileNet, debemos proporcionar un número de punto flotante para cada canal donde el
   * valor esté entre 0 y 1. Para hacer esto, enmascaramos cada canal de color como antes, pero luego dividimos
   * cada valor resultante entre 255.f.
   * En efecto el método "addPixelValue" normaliza los pixeles escalandalos en [0.0,1.0] y convierte la imagen
   * a punto flotante.
   * http://borg.csueastbay.edu/~grewe/CS663/Exercises/ExerciseA4Before.html
   * */
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.f); //Normaliza canal Rojo (R)
        imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.f); //Normaliza canal Verde (G)
        imgData.putFloat((pixelValue & 0xFF) / 255.f); //Normaliza canal Azul (B)
        // >> : Operador a nivel de bits (shifts the bits to right) Cambia los bits a la derecha
        // https://stackoverflow.com/questions/6126439/what-does-0xff-do
    }

    //Otros: Convertir pixeles a escala de grises:
   /*
    protected void addPixelValue(int pixelValue) {
        float mean = (((pixelValue >> 16) & 0xFF) + ((pixelValue >> 8) & 0xFF) +
                (pixelValue & 0xFF)) / 3.0f;
        imgData.putFloat(mean / 127.5f - 1.0f);
    }
    */


    private int argmax(float[] probs){ //[0.76111, 0.50311, 0.30111
        Log.d("array", "=> "+Arrays.toString(probs));
        int maxIds = -1;
        float maxProb = 0.0f;
        for (int i=0; i<probs.length; i++){
            if(probs[i] > maxProb){
                maxProb = probs[i]; //0.76111
                maxIds = i; //0
            }
        }
        return maxIds;
    }

}