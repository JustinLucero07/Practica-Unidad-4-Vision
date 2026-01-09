#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Constantes que permiten especificar tamaño de la imagen como los
// umbrales de confianza del modelo
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Estructura de datos que almacenará las detecciones
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

vector<string> load_class_list() {
    vector<string> class_list;
    ifstream ifs("classes.txt");
    string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// Se define el tamaño (cuadrado) con el que trabaja YOLO como entrada
Mat format_yolo(Mat source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

int main(int argc, char **argv) {
    vector<string> class_list = load_class_list();

    // 1. Cargamos la red YOLO exportada desde el Cuaderno de Google Colab (con el paquete Ultralytics)
    Net net = readNet("yolo12n.onnx");
    
    // Activamos la ejecución del modelo en CUDA
    //net.setPreferableBackend(DNN_BACKEND_CUDA);
    //net.setPreferableTarget(DNN_TARGET_CUDA_FP16);

    // 2. Definimos la fuente desde donde se cargará el vídeo

    VideoCapture cap("Moscow.mp4"); 

    if (!cap.isOpened()) {
        cout << "Error, no se puede abrir la fuente de vídeo" << endl;
        return -1;
    }

    Mat frame;
    vector<Mat> outputs;
    namedWindow("Detección Objetos YOLOv12", WINDOW_AUTOSIZE);
    Mat modelInput;
    Mat blob;
    Mat output_data;
    float *data;
    int rows = 0;
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    float x_factor;
    float y_factor;
    float *row_ptr;  
    Mat scores;
    Point class_id;
    double max_class_score;
    float x, y, w, h;
    int left, top, width, height;
    Detection result;
    vector<int> nms_result;
    string label;
    int baseLine;
    Size labelSize;

    double lastTime;
    double currentTime;
    double fps;


    while (3==3) {
        cap >> frame;

        if (frame.empty())
            break;
        
        class_ids.clear();
        confidences.clear();
        boxes.clear();

        lastTime = (double) getTickCount();

        // 3. Pre-procesamiento de la imagen
        modelInput = format_yolo(frame);
        
        // Generamos el blob de datos a partir de la imagen        
        // Normalizando de 0 a 1 y con un tamaño base
        blobFromImage(modelInput, blob, 1.0/255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

        // Alimentamos a la red con la imagen normalizada
        net.setInput(blob);

        // 4. Inferencia
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 5. Obtenemos la salida que devuelve YOLO en forma de matriz
        output_data = outputs[0];
        
        // YOLO normalmente devuelve los datos como un tensor con la siguiente estructura:
        // 1 x Canales x Anchors [1, 84, 8400], donde:
        // 1 Es el batch, en este caso, una sola imagen
        // 84 Es el número de atributos por box (4 coordenadas x, y, h, w, + 80 probabilidades por clase del COCO Dataset) 
        // 8400 Son las detecciones (predicciones) que hace por defecto YOLO para una imagen de 640x640
        
        // Con Reshape [1, 84, 8400] -> [84, 8400] eliminamos el batch, similar a squeezee de PyTorch
        if (output_data.dims > 2) {
            // Con esto obtenemos una matrix de 84 x 8400
            output_data = output_data.reshape(0, output_data.size[1]);
        }

        // Este método es clave, ya que orginalmente los datos están organizados por filas
        // es decir, la fila 0 contiene las coordenas X de todos los objetos detectados
        // la fila 1 las coordenadas Y, y así sucesivamente
        // Con el método transpose re-organizamos en 8400 detecciones (filas) con sus 
        // 84 características (x, y, h, w, y probabilidades de las 80 clases)
        transpose(output_data, output_data); 
        
        // Obtenemos el puntero de datos de las detecciones, esto es mucho más rápido que usar el método .at<>(i,j)
        data =  (float *) output_data.data;
        rows =  output_data.rows; // 8400

        // Factor de escala, debemos recordar que YOLO usa imágenes de 640x640
        // por ello, debemos obtener este factor para que cuando se dibujen los
        // rectángulos se realice correctamente en la imagen original
        x_factor = modelInput.cols / INPUT_WIDTH;
        y_factor = modelInput.rows / INPUT_HEIGHT;

        for (int i = 0; i < rows; ++i) {
            row_ptr = output_data.ptr<float>(i);
            
            // El umbral de confianza inicia en el índice 4 (0-3 son las coords. del rectangulo donde se detecta el objeto)
            scores = Mat(1, class_list.size(), CV_32FC1, row_ptr + 4);
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score >= CONFIDENCE_THRESHOLD) {
                x = row_ptr[0];
                y = row_ptr[1];
                w = row_ptr[2];
                h = row_ptr[3];

                left = int((x - 0.5 * w) * x_factor);
                top = int((y - 0.5 * h) * y_factor);
                width = int(w * x_factor);
                height = int(h * y_factor);

                boxes.push_back(Rect(left, top, width, height));
                class_ids.push_back(class_id.x);
                confidences.push_back((float)max_class_score);
            }
        }

        // 6. Etapa de Non-Maximum Suppression
        nms_result.clear();
        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

        // 7. Dibujamos los rectángulos con las detecciones
        for (int idx : nms_result) {
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];

            rectangle(frame, result.box, Scalar(0, 255, 0), 2);

            label = class_list[result.class_id] + ": " + to_string(result.confidence).substr(0, 4);
            
            // Dibujamos las etiquetas
            labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Point(result.box.x, result.box.y - labelSize.height - 5),
                      Point(result.box.x + labelSize.width, result.box.y), Scalar(0, 255, 0), FILLED);
            putText(frame, label, Point(result.box.x, result.box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        currentTime = (double) getTickCount();
        fps = getTickFrequency() / (currentTime - lastTime);

        lastTime = currentTime;
        putText(frame, "FPS: " + to_string(int(fps)), Point(20, 50), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        
        imshow("Detección Objetos YOLOv12", frame);

        if (waitKey(1) == 27) 
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}