#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

double tiempo_cpu = 0;
double tiempo_cpu_gpu = 0;  
double tiempo_gpu_only = 0;

double cpu_gaussian = 0, cpu_morph = 0, cpu_canny = 0, cpu_hist = 0;

double hybrid_gaussian = 0, hybrid_morph = 0, hybrid_canny = 0, hybrid_hist = 0;

double gpu_gaussian = 0, gpu_morph = 0, gpu_canny = 0, gpu_hist = 0;

double getTime() {
    return (double)getTickCount() / getTickFrequency() * 1000.0;
}

// PIPELINE 1: CPU 
Mat procesarCPU(Mat frame) {
    double inicio_total = getTime();
    Mat gray, blur, morph, edges, eq, resultado;
    
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    // 1. Suavizado - Filtro Gaussiano
    double t1 = getTime();
    GaussianBlur(gray, blur, Size(5, 5), 1.5);
    cpu_gaussian = getTime() - t1;
    
    // 2. Operaciones morfológicas
    double t2 = getTime();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(blur, morph, kernel);
    dilate(morph, morph, kernel);
    cpu_morph = getTime() - t2;
    
    // 3. Detección de bordes - Canny
    double t3 = getTime();
    Canny(morph, edges, 50, 150);
    cpu_canny = getTime() - t3;
    
    // 4. Ecualización del histograma
    double t4 = getTime();
    equalizeHist(gray, eq);
    cpu_hist = getTime() - t4;
    
    hconcat(edges, eq, resultado);
    cvtColor(resultado, resultado, COLOR_GRAY2BGR);
    
    tiempo_cpu = getTime() - inicio_total;
    return resultado;
}

// PIPELINE 2: CPU ↔ GPU (HÍBRIDO INEFICIENTE)
Mat procesarCPU_GPU_Hybrid(Mat frame) {
    double inicio_total = getTime();
    Mat gray, blur, morph, edges, eq, resultado;
    
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    // 1. Gaussian Blur
    double t1 = getTime();
    cuda::GpuMat d_gray, d_blur;
    d_gray.upload(gray);  
    auto gaussian = cuda::createGaussianFilter(d_gray.type(), d_gray.type(), 
                                                Size(5, 5), 1.5);
    gaussian->apply(d_gray, d_blur);
    d_blur.download(blur);  
    hybrid_gaussian = getTime() - t1;
    
    // 2. Morfología
    double t2 = getTime();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    cuda::GpuMat d_morph_in, d_morph_out;
    d_morph_in.upload(blur);  
    auto eroder = cuda::createMorphologyFilter(MORPH_ERODE, d_morph_in.type(), kernel);
    eroder->apply(d_morph_in, d_morph_out);
    auto dilater = cuda::createMorphologyFilter(MORPH_DILATE, d_morph_out.type(), kernel);
    dilater->apply(d_morph_out, d_morph_out);
    d_morph_out.download(morph);  
    hybrid_morph = getTime() - t2;
    
    // 3. Canny
    double t3 = getTime();
    cuda::GpuMat d_canny_in, d_edges;
    d_canny_in.upload(morph);  
    auto canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_canny_in, d_edges);
    d_edges.download(edges);  
    hybrid_canny = getTime() - t3;
    
    // 4. Histograma
    double t4 = getTime();
    cuda::GpuMat d_hist_in, d_eq;
    d_hist_in.upload(gray);  
    cuda::equalizeHist(d_hist_in, d_eq);
    d_eq.download(eq);  
    hybrid_hist = getTime() - t4;
    
    hconcat(edges, eq, resultado);
    cvtColor(resultado, resultado, COLOR_GRAY2BGR);
    
    tiempo_cpu_gpu = getTime() - inicio_total;
    return resultado;
}

// PIPELINE 3: GPU-ONLY 
Mat procesarGPU_Only(Mat frame) {
    double inicio_total = getTime();
    
    cuda::GpuMat d_frame, d_gray, d_blur, d_morph, d_edges, d_eq;
    d_frame.upload(frame);
    
    cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
    
    // 1. Gaussian
    double t1 = getTime();
    auto gaussian = cuda::createGaussianFilter(d_gray.type(), d_gray.type(), 
                                                Size(5, 5), 1.5);
    gaussian->apply(d_gray, d_blur);
    gpu_gaussian = getTime() - t1;
    
    // 2. Morfología
    double t2 = getTime();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    auto eroder = cuda::createMorphologyFilter(MORPH_ERODE, d_blur.type(), kernel);
    eroder->apply(d_blur, d_morph);
    auto dilater = cuda::createMorphologyFilter(MORPH_DILATE, d_morph.type(), kernel);
    dilater->apply(d_morph, d_morph);
    gpu_morph = getTime() - t2;
    
    // 3. Canny
    double t3 = getTime();
    auto canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_morph, d_edges);
    gpu_canny = getTime() - t3;
    
    // 4. Histograma
    double t4 = getTime();
    cuda::equalizeHist(d_gray, d_eq);
    gpu_hist = getTime() - t4;
    
    Mat edges, eq, resultado;
    d_edges.download(edges);
    d_eq.download(eq);
    
    hconcat(edges, eq, resultado);
    cvtColor(resultado, resultado, COLOR_GRAY2BGR);
    
    tiempo_gpu_only = getTime() - inicio_total;
    return resultado;
}

void mostrarInfo(Mat& img, string texto, double tiempo, Scalar color) {
    putText(img, texto, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    putText(img, to_string((int)tiempo) + " ms", Point(10, 53), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(img, to_string((int)(1000.0/tiempo)) + " FPS", Point(10, 70), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
}

int main(int argc, char** argv) {
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cout << "ERROR: No hay GPU con CUDA disponible" << endl;
        return -1;
    }
    
    cout << "\n==================================" << endl;
    cout << "GPU CUDA detectada:" << endl;
    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    cout << "==================================\n" << endl;
    
    VideoCapture cap;
    if (argc > 1) {
        cap.open(argv[1]);
        cout << "Procesando video: " << argv[1] << endl;
    } else {
        cap.open(0);
        cout << "Usando camara web" << endl;
    }
    
    if (!cap.isOpened()) {
        cout << "ERROR: No se pudo abrir video/camara" << endl;
        return -1;
    }
    
    cout << "\nMostrando 4 ventanas:" << endl;
    cout << "  - Original" << endl;
    cout << "  - Pipeline CPU (todo en RAM)" << endl;
    cout << "  - Pipeline CPU↔GPU (hibrido - muchas transferencias)" << endl;
    cout << "  - Pipeline GPU-only (optimizado)" << endl;
    cout << "\nPresiona ESC para ver el analisis completo\n" << endl;
    
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("CPU", WINDOW_NORMAL);
    namedWindow("CPU↔GPU", WINDOW_NORMAL);
    namedWindow("GPU-only", WINDOW_NORMAL);
    
    resizeWindow("Original", 400, 320);
    resizeWindow("CPU", 400, 320);
    resizeWindow("CPU↔GPU", 400, 320);
    resizeWindow("GPU-only", 400, 320);
    
    moveWindow("Original", 0, 0);
    moveWindow("CPU", 420, 0);
    moveWindow("CPU↔GPU", 840, 0);
    moveWindow("GPU-only", 1260, 0);
    
    Mat frame;
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        resize(frame, frame, Size(640, 480));
        
        Mat resultado_cpu = procesarCPU(frame);
        Mat resultado_hybrid = procesarCPU_GPU_Hybrid(frame);
        Mat resultado_gpu = procesarGPU_Only(frame);
        
        Mat frame_original = frame.clone();
        putText(frame_original, "ORIGINAL", Point(10, 35), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        
        mostrarInfo(resultado_cpu, "CPU", tiempo_cpu, Scalar(100, 100, 255));
        mostrarInfo(resultado_hybrid, "CPU<->GPU", tiempo_cpu_gpu, Scalar(255, 165, 0));
        mostrarInfo(resultado_gpu, "GPU-only", tiempo_gpu_only, Scalar(100, 255, 100));
        
        imshow("Original", frame_original);
        imshow("CPU", resultado_cpu);
        imshow("CPU↔GPU", resultado_hybrid);
        imshow("GPU-only", resultado_gpu);
        
        frame_count++;
        
        if (waitKey(1) == 27) break;
    }
    
    cout << "\n\n" << endl;
    cout << "============================================================" << endl;
    cout << "               ANALISIS FINAL DE RESULTADOS" << endl;
    cout << "============================================================" << endl;
    cout << "Configuracion:" << endl;
    cout << "  - Frames procesados: " << frame_count << endl;
    cout << "  - Resolucion: 640x480 pixeles" << endl;
    cout << "  - Operaciones: Gaussian, Morfologia, Canny, Histograma" << endl;
    
    cout << "\n------------------------------------------------------------" << endl;
    cout << "           COMPARACION DE LOS 3 PIPELINES" << endl;
    cout << "------------------------------------------------------------" << endl;
    
    cout << "\n1. TIEMPOS TOTALES PROMEDIO:" << endl;
    cout << "   Pipeline CPU:           " << fixed << setprecision(2) << tiempo_cpu << " ms" << endl;
    cout << "   Pipeline CPU↔GPU:       " << tiempo_cpu_gpu << " ms" << endl;
    cout << "   Pipeline GPU-only:      " << tiempo_gpu_only << " ms" << endl;
    
    cout << "\n2. TABLA COMPARATIVA POR OPERACION:" << endl;
    cout << "   +------------------+----------+----------+----------+" << endl;
    cout << "   | Operacion        | CPU (ms) | CPU↔GPU  | GPU-only |" << endl;
    cout << "   +------------------+----------+----------+----------+" << endl;
    cout << "   | Gaussian Blur    | " << setw(8) << cpu_gaussian 
         << " | " << setw(8) << hybrid_gaussian << " | " << setw(8) << gpu_gaussian << " |" << endl;
    cout << "   | Morfologia       | " << setw(8) << cpu_morph 
         << " | " << setw(8) << hybrid_morph << " | " << setw(8) << gpu_morph << " |" << endl;
    cout << "   | Canny Edges      | " << setw(8) << cpu_canny 
         << " | " << setw(8) << hybrid_canny << " | " << setw(8) << gpu_canny << " |" << endl;
    cout << "   | Histograma       | " << setw(8) << cpu_hist 
         << " | " << setw(8) << hybrid_hist << " | " << setw(8) << gpu_hist << " |" << endl;
    cout << "   +------------------+----------+----------+----------+" << endl;
    cout << "   | TOTAL            | " << setw(8) << tiempo_cpu 
         << " | " << setw(8) << tiempo_cpu_gpu << " | " << setw(8) << tiempo_gpu_only << " |" << endl;
    cout << "   +------------------+----------+----------+----------+" << endl;
    
    cout << "\n4. FPS (Frames Por Segundo):" << endl;
    cout << "   CPU:         " << (int)(1000.0/tiempo_cpu) << " fps" << endl;
    cout << "   CPU↔GPU:     " << (int)(1000.0/tiempo_cpu_gpu) << " fps" << endl;
    cout << "   GPU-only:    " << (int)(1000.0/tiempo_gpu_only) << " fps" << endl;

}
