#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const bool USE_GPU_SR = true;

void configure_backend(Net& net, bool use_gpu)
{
    if (use_gpu)
    {
        try
        {
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
            cout << "[SR] GPU activada" << endl;
        }
        catch (...)
        {
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            cout << "[SR] CUDA no disponible, usando CPU" << endl;
        }
    }
    else
    {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "[SR] Usando CPU" << endl;
    }
}

int main()
{
    ifstream f("RealESR-AnimeVideo-v3_x4.onnx");
    if (!f.good())
    {
        cout << "Modelo ONNX no encontrado" << endl;
        return -1;
    }

    Net sr_net = readNet("RealESR-AnimeVideo-v3_x4.onnx");
    configure_backend(sr_net, USE_GPU_SR);

    VideoCapture cap("Moscow.mp4");
    if (!cap.isOpened())
    {
        cout << "Error abriendo video" << endl;
        return -1;
    }

    namedWindow("Super Resolucion", WINDOW_AUTOSIZE);

    Mat frame, sr_frame;
    vector<Mat> outputs;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Tamaño original del video
        Size original_size(frame.cols, frame.rows);

        double t0 = getTickCount();

        Mat rgb;
        cvtColor(frame, rgb, COLOR_BGR2RGB);

        Mat input;
        rgb.convertTo(input, CV_32FC3, 1.0 / 255.0);

        Mat blob = blobFromImage(
            input,
            1.0,
            Size(),          // No forzar tamaño
            Scalar(),
            false,
            false
        );

        sr_net.setInput(blob);
        sr_net.forward(outputs);

        Mat out = outputs[0];
        int h = out.size[2];
        int w = out.size[3];

        vector<Mat> ch(3);
        for (int c = 0; c < 3; c++)
        {
            ch[c] = Mat(h, w, CV_32F, out.ptr<float>(0, c));
        }

        merge(ch, sr_frame);
        sr_frame.convertTo(sr_frame, CV_8UC3, 255.0);
        cvtColor(sr_frame, sr_frame, COLOR_RGB2BGR);

        // Volver al tamaño original (854x480)
        resize(sr_frame, sr_frame, original_size, 0, 0, INTER_AREA);

        double fps = getTickFrequency() / (getTickCount() - t0);

        putText(
            sr_frame,
            "Super Resolucion (" + string(USE_GPU_SR ? "GPU" : "CPU") + ") - " +
                to_string((int)fps) + " FPS",
            Point(25, 45),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0, 0, 255),
            2
        );

        imshow("Super Resolucion", sr_frame);

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
