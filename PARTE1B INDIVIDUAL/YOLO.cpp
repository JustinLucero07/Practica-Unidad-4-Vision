#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/* ===================== CONFIGURACIÓN GPU/CPU ===================== */
const bool USE_GPU_YOLO = false;   // true = GPU, false = CPU
const bool USE_GPU_SR = false;     // true = GPU, false = CPU

/* ===================== CONSTANTES ===================== */
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

/* ===================== ESTRUCTURA ===================== */
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

/* ===================== CLASES ===================== */
vector<string> load_class_list() {
    vector<string> class_list;
    ifstream ifs("classes.txt");
    string line;
    while (getline(ifs, line))
        class_list.push_back(line);
    return class_list;
}

/* ===================== YOLO INPUT ===================== */
Mat format_yolo(Mat source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

/* ===================== CONFIGURAR BACKEND ===================== */
void configure_backend(Net& net, bool use_gpu, const string& name) {
    if (use_gpu) {
        cout << "[" << name << "] Intentando usar GPU (CUDA)..." << endl;
        try {
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
            cout << "[" << name << "] GPU activada correctamente" << endl;
        } catch (const exception& e) {
            cout << "[" << name << "] GPU no disponible, usando CPU" << endl;
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }
    } else {
        cout << "[" << name << "] Usando CPU" << endl;
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

int main() {

    cout << "========================================" << endl;
    cout << "   CONFIGURACION DE PROCESAMIENTO" << endl;
    cout << "========================================" << endl;
    cout << "YOLO GPU: " << (USE_GPU_YOLO ? "ACTIVADA" : "DESACTIVADA") << endl;
    cout << "SR GPU:   " << (USE_GPU_SR ? "ACTIVADA" : "DESACTIVADA") << endl;
    cout << "========================================" << endl;

    vector<string> class_list = load_class_list();

    /* ===================== YOLO ===================== */
    Net net = readNet("yolo12n.onnx");
    configure_backend(net, USE_GPU_YOLO, "YOLO");

    /* ===================== Real-ESRGAN x4 (ONNX) ===================== */
    Net sr_net;
    bool sr_available = false;
    
    try {
        // Verificar si el archivo existe
        ifstream sr_file("RealESRGAN_x4plus.fp16.onnx");
        if (!sr_file.good()) {
            cout << "[SR] WARNING: Archivo RealESR-AnimeVideo-v3_x4.onnx no encontrado" << endl;
            cout << "[SR] Descargalo desde tu fuente o usa otro modelo ONNX" << endl;
            cout << "[SR] Continuando sin Super Resolution..." << endl;
        } else {
            sr_net = readNet("RealESR-AnimeVideo-v3_x4.onnx");
            configure_backend(sr_net, USE_GPU_SR, "Real-ESRGAN");
            sr_available = true;
            cout << "[SR] Real-ESRGAN cargada correctamente" << endl;
        }
    } catch (const exception& e) {
        cout << "[SR] Error cargando Real-ESRGAN: " << e.what() << endl;
        cout << "[SR] Continuando sin Super Resolution..." << endl;
        sr_available = false;
    }

    VideoCapture cap("Moscow.mp4");
    if (!cap.isOpened()) {
        cout << "Error abriendo video\n";
        return -1;
    }

    // Mostrar info del video
    int video_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double video_fps = cap.get(CAP_PROP_FPS);
    
    cout << "========================================" << endl;
    cout << "Video: " << video_width << "x" << video_height << " @ " << video_fps << " FPS" << endl;
    cout << "========================================" << endl;

    namedWindow("YOLOv12", WINDOW_AUTOSIZE);
    if (sr_available) {
        namedWindow("Real-ESRGAN x4", WINDOW_AUTOSIZE);
    }

    Mat frame, sr_frame;
    vector<Mat> outputs;
    vector<Mat> sr_outputs;

    double fps_yolo = 0.0;
    double fps_sr = 0.0;
    
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        frame_count++;

        /* ================= YOLO ================= */
        double t0 = getTickCount();

        Mat modelInput = format_yolo(frame);
        Mat blob;
        blobFromImage(modelInput, blob, 1.0/255.0,
                      Size(INPUT_WIDTH, INPUT_HEIGHT),
                      Scalar(), true, false);

        net.setInput(blob);
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        Mat output_data = outputs[0];
        if (output_data.dims > 2)
            output_data = output_data.reshape(0, output_data.size[1]);
        transpose(output_data, output_data);

        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;

        float x_factor = modelInput.cols / INPUT_WIDTH;
        float y_factor = modelInput.rows / INPUT_HEIGHT;

        for (int i = 0; i < output_data.rows; i++) {
            float* row = output_data.ptr<float>(i);
            Mat scores(1, class_list.size(), CV_32FC1, row + 4);

            Point class_id;
            double max_score;
            minMaxLoc(scores, 0, &max_score, 0, &class_id);

            if (max_score >= CONFIDENCE_THRESHOLD) {
                int x = (row[0] - 0.5 * row[2]) * x_factor;
                int y = (row[1] - 0.5 * row[3]) * y_factor;
                int w = row[2] * x_factor;
                int h = row[3] * y_factor;

                boxes.push_back(Rect(x, y, w, h));
                confidences.push_back(max_score);
                class_ids.push_back(class_id.x);
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        Mat yolo_view = frame.clone();
        for (int idx : indices) {
            rectangle(yolo_view, boxes[idx], Scalar(0,255,0), 2);
            putText(yolo_view, class_list[class_ids[idx]],
                    Point(boxes[idx].x, boxes[idx].y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
        }

        fps_yolo = getTickFrequency() / (getTickCount() - t0);

        /* ================= Real-ESRGAN x4 (ONNX) ================= */
        if (sr_available) {
            double t1 = getTickCount();
            
            try {
                // Convertir BGR a RGB
                Mat frame_rgb;
                cvtColor(frame, frame_rgb, COLOR_BGR2RGB);
                
                // Convertir a float32 y normalizar [0, 1]
                Mat frame_float;
                frame_rgb.convertTo(frame_float, CV_32FC3, 1.0/255.0);
                
                // Crear blob con formato NCHW (1, 3, H, W)
                Mat sr_blob = blobFromImage(frame_float, 1.0, frame.size(), 
                                             Scalar(), false, false);
                
                sr_net.setInput(sr_blob);
                sr_net.forward(sr_outputs, sr_net.getUnconnectedOutLayersNames());
                
                // Obtener salida
                Mat sr_output = sr_outputs[0];
                
                // Convertir de NCHW (1, 3, H, W) a HWC
                vector<Mat> channels(3);
                int h = sr_output.size[2];
                int w = sr_output.size[3];
                
                for (int c = 0; c < 3; c++) {
                    channels[c] = Mat(h, w, CV_32F, sr_output.ptr<float>(0, c));
                }
                
                merge(channels, sr_frame);
                
                // Desnormalizar y convertir a uint8
                sr_frame.convertTo(sr_frame, CV_8UC3, 255.0);
                
                // Convertir RGB de vuelta a BGR para mostrar
                cvtColor(sr_frame, sr_frame, COLOR_RGB2BGR);
                
                // Real-ESRGAN x4 produce 4x el tamaño, redimensionar al original
                resize(sr_frame, sr_frame, frame.size());
                
                fps_sr = getTickFrequency() / (getTickCount() - t1);
            } catch (const exception& e) {
                cout << "[SR] Error procesando frame: " << e.what() << endl;
                sr_frame = frame.clone();
                fps_sr = 0;
            }
        }

        // Mostrar backend usado
        string yolo_backend = USE_GPU_YOLO ? "GPU" : "CPU";
        string sr_backend = USE_GPU_SR ? "GPU" : "CPU";

        putText(yolo_view, "YOLO (" + yolo_backend + "): " + to_string((int)fps_yolo) + " FPS",
                Point(20,40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255), 2);

        imshow("YOLOv12", yolo_view);
        
        if (sr_available) {
            putText(sr_frame, "Real-ESRGAN (" + sr_backend + "): " + to_string((int)fps_sr) + " FPS",
                    Point(20,40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
            imshow("Real-ESRGAN x4", sr_frame);
        }
        
        // Mostrar progreso cada 30 frames
        if (frame_count % 30 == 0) {
            cout << "Frame " << frame_count 
                 << " | YOLO: " << (int)fps_yolo << " FPS";
            if (sr_available) {
                cout << " | SR: " << (int)fps_sr << " FPS";
            }
            cout << endl;
        }

        if (waitKey(1) == 27) break;
    }

    cout << "========================================" << endl;
    cout << "Procesamiento completado: " << frame_count << " frames" << endl;
    cout << "========================================" << endl;

    cap.release();
    destroyAllWindows();
    return 0;
}