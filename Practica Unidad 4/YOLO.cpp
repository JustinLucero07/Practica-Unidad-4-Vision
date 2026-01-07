#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn_superres.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace cv::dnn_superres;
using namespace std;

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

int main() {

    vector<string> class_list = load_class_list();

    /* ===================== YOLO ===================== */
    Net net = readNet("yolo12n.onnx");

    // ===== GPU =====
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);

    // ===== CPU =====
    //net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(DNN_TARGET_CPU);

    /* ===================== SUPER RES ===================== */
    DnnSuperResImpl sr;
    sr.readModel("ESPCN_x4.pb");
    sr.setModel("espcn", 4);

    VideoCapture cap("Moscow.mp4");
    if (!cap.isOpened()) {
        cout << "Error abriendo video\n";
        return -1;
    }

    namedWindow("YOLOv12", WINDOW_AUTOSIZE);
    namedWindow("Super Resolucion x4", WINDOW_AUTOSIZE);

    Mat frame, sr_frame;
    vector<Mat> outputs;

    double fps_yolo = 0.0;
    double fps_sr = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

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

        /* ================= SUPER RES ================= */
        double t1 = getTickCount();
        sr.upsample(frame, sr_frame);
        resize(sr_frame, sr_frame, frame.size());
        fps_sr = getTickFrequency() / (getTickCount() - t1);

        putText(yolo_view, "FPS YOLO: " + to_string((int)fps_yolo),
                Point(20,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);

        putText(sr_frame, "FPS SR: " + to_string((int)fps_sr),
                Point(20,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0), 2);

        imshow("YOLOv12", yolo_view);
        imshow("Super Resolucion x4", sr_frame);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
