#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// Estructura para almacenar tiempos de procesamiento
struct ProcessingTimes {
    double gaussian;
    double morphology;
    double canny;
    double histogram;
    double total;
    int transfers; // Número de transferencias CPU-GPU
};

// Función para medir tiempo
class Timer {
    chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(chrono::high_resolution_clock::now()) {}
    double elapsed() {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end - start).count();
    }
};

// Pipeline CPU
ProcessingTimes processCPU(const Mat& input, Mat& output) {
    ProcessingTimes times;
    times.transfers = 0; // CPU no tiene transferencias
    
    Timer totalTimer;
    Mat gray, blurred, morphed, edges, equalized;
    
    // Convertir a escala de grises
    cvtColor(input, gray, COLOR_BGR2GRAY);
    
    // 1. Suavizado - Filtro Gaussiano
    Timer timer;
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);
    times.gaussian = timer.elapsed();
    
    // 2. Operaciones morfológicas - Erosión seguida de Dilatación
    timer = Timer();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(blurred, morphed, kernel);
    dilate(morphed, morphed, kernel);
    times.morphology = timer.elapsed();
    
    // 3. Detección de bordes - Canny
    timer = Timer();
    Canny(morphed, edges, 50, 150);
    times.canny = timer.elapsed();
    
    // 4. Ecualización del histograma
    timer = Timer();
    equalizeHist(gray, equalized);
    times.histogram = timer.elapsed();
    
    times.total = totalTimer.elapsed();
    
    // Combinar resultados para visualización
    Mat combined;
    hconcat(vector<Mat>{edges, equalized}, combined);
    cvtColor(combined, output, COLOR_GRAY2BGR);
    
    return times;
}

// Pipeline CPU ↔ GPU (INEFICIENTE - múltiples transferencias)
ProcessingTimes processGPU_Inefficient(const Mat& input, Mat& output) {
    ProcessingTimes times;
    times.transfers = 0;
    
    Timer totalTimer;
    cuda::GpuMat d_input, d_gray, d_blurred, d_morphed, d_edges, d_equalized;
    Mat gray, blurred, morphed, edges, equalized;
    
    // Convertir a escala de grises en CPU
    cvtColor(input, gray, COLOR_BGR2GRAY);
    
    // 1. Suavizado - Filtro Gaussiano
    Timer timer;
    d_input.upload(gray); times.transfers++; // CPU → GPU
    auto gaussian = cuda::createGaussianFilter(d_input.type(), d_input.type(), 
                                                Size(5, 5), 1.5);
    gaussian->apply(d_input, d_blurred);
    d_blurred.download(blurred); times.transfers++; // GPU → CPU
    times.gaussian = timer.elapsed();
    
    // 2. Operaciones morfológicas
    timer = Timer();
    d_input.upload(blurred); times.transfers++; // CPU → GPU
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    auto morphFilter = cuda::createMorphologyFilter(MORPH_ERODE, d_input.type(), kernel);
    morphFilter->apply(d_input, d_morphed);
    morphFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_morphed.type(), kernel);
    morphFilter->apply(d_morphed, d_morphed);
    d_morphed.download(morphed); times.transfers++; // GPU → CPU
    times.morphology = timer.elapsed();
    
    // 3. Detección de bordes - Canny
    timer = Timer();
    d_input.upload(morphed); times.transfers++; // CPU → GPU
    auto canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_input, d_edges);
    d_edges.download(edges); times.transfers++; // GPU → CPU
    times.canny = timer.elapsed();
    
    // 4. Ecualización del histograma
    timer = Timer();
    d_input.upload(gray); times.transfers++; // CPU → GPU
    cuda::equalizeHist(d_input, d_equalized);
    d_equalized.download(equalized); times.transfers++; // GPU → CPU
    times.histogram = timer.elapsed();
    
    times.total = totalTimer.elapsed();
    
    // Combinar resultados
    Mat combined;
    hconcat(vector<Mat>{edges, equalized}, combined);
    cvtColor(combined, output, COLOR_GRAY2BGR);
    
    return times;
}

// Pipeline GPU-only (EFICIENTE - una sola transferencia)
ProcessingTimes processGPU_Optimized(const Mat& input, Mat& output) {
    ProcessingTimes times;
    times.transfers = 2; // Solo upload y download
    
    Timer totalTimer;
    cuda::GpuMat d_input, d_gray, d_blurred, d_morphed, d_edges, d_equalized;
    
    // ===== UPLOAD ÚNICO =====
    d_input.upload(input); // CPU → GPU (única transferencia de entrada)
    
    // Convertir a escala de grises en GPU
    cuda::cvtColor(d_input, d_gray, COLOR_BGR2GRAY);
    
    // 1. Suavizado - Filtro Gaussiano (GPU)
    Timer timer;
    auto gaussian = cuda::createGaussianFilter(d_gray.type(), d_gray.type(), 
                                                Size(5, 5), 1.5);
    gaussian->apply(d_gray, d_blurred);
    times.gaussian = timer.elapsed();
    
    // 2. Operaciones morfológicas (GPU)
    timer = Timer();
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    auto morphErode = cuda::createMorphologyFilter(MORPH_ERODE, d_blurred.type(), kernel);
    morphErode->apply(d_blurred, d_morphed);
    auto morphDilate = cuda::createMorphologyFilter(MORPH_DILATE, d_morphed.type(), kernel);
    morphDilate->apply(d_morphed, d_morphed);
    times.morphology = timer.elapsed();
    
    // 3. Detección de bordes - Canny (GPU)
    timer = Timer();
    auto canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_morphed, d_edges);
    times.canny = timer.elapsed();
    
    // 4. Ecualización del histograma (GPU)
    timer = Timer();
    cuda::equalizeHist(d_gray, d_equalized);
    times.histogram = timer.elapsed();
    
    // ===== DOWNLOAD ÚNICO =====
    Mat edges, equalized;
    d_edges.download(edges); // GPU → CPU
    d_equalized.download(equalized); // GPU → CPU
    
    times.total = totalTimer.elapsed();
    
    // Combinar resultados
    Mat combined;
    hconcat(vector<Mat>{edges, equalized}, combined);
    cvtColor(combined, output, COLOR_GRAY2BGR);
    
    return times;
}

// Función para imprimir resultados
void printResults(const string& method, const ProcessingTimes& times) {
    cout << "\n========== " << method << " ==========\n";
    cout << "Gaussian Blur:     " << fixed << setprecision(2) << times.gaussian << " ms\n";
    cout << "Morphology:        " << times.morphology << " ms\n";
    cout << "Canny Detection:   " << times.canny << " ms\n";
    cout << "Histogram Eq:      " << times.histogram << " ms\n";
    cout << "TOTAL TIME:        " << times.total << " ms\n";
    cout << "CPU-GPU Transfers: " << times.transfers << "\n";
    cout << "=====================================\n";
}

int main(int argc, char** argv) {
    // Verificar disponibilidad de CUDA
    int deviceCount = cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        cerr << "Error: No hay dispositivos CUDA disponibles!\n";
        return -1;
    }
    
    cout << "Dispositivos CUDA encontrados: " << deviceCount << endl;
    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    
    // Cargar imagen o usar cámara
    string imagePath = (argc > 1) ? argv[1] : "";
    VideoCapture cap;
    Mat frame;
    
    if (!imagePath.empty()) {
        frame = imread(imagePath);
        if (frame.empty()) {
            cerr << "Error: No se pudo cargar la imagen: " << imagePath << endl;
            return -1;
        }
    } else {
        cap.open(0); // Abrir cámara
        if (!cap.isOpened()) {
            cerr << "Error: No se pudo abrir la cámara\n";
            return -1;
        }
        cap.read(frame);
    }
    
    // Redimensionar para procesamiento más rápido
    resize(frame, frame, Size(640, 480));
    
    cout << "\n====================================\n";
    cout << "  ANALISIS DE RENDIMIENTO\n";
    cout << "====================================\n";
    cout << "Imagen: " << frame.cols << "x" << frame.rows << " pixels\n";
    
    // Variables para resultados
    Mat cpu_result, gpu_inefficient_result, gpu_optimized_result;
    ProcessingTimes cpu_times, gpu_ineff_times, gpu_opt_times;
    
    // Calentar GPU (warmup)
    cout << "\nCalentando GPU...\n";
    for (int i = 0; i < 5; i++) {
        processGPU_Optimized(frame, gpu_optimized_result);
    }
    
    // Ejecutar pruebas múltiples veces para promediar
    const int iterations = 10;
    cout << "\nEjecutando " << iterations << " iteraciones por método...\n";
    
    ProcessingTimes cpu_avg = {0}, gpu_ineff_avg = {0}, gpu_opt_avg = {0};
    
    for (int i = 0; i < iterations; i++) {
        // CPU
        cpu_times = processCPU(frame, cpu_result);
        cpu_avg.gaussian += cpu_times.gaussian;
        cpu_avg.morphology += cpu_times.morphology;
        cpu_avg.canny += cpu_times.canny;
        cpu_avg.histogram += cpu_times.histogram;
        cpu_avg.total += cpu_times.total;
        
        // GPU Ineficiente
        gpu_ineff_times = processGPU_Inefficient(frame, gpu_inefficient_result);
        gpu_ineff_avg.gaussian += gpu_ineff_times.gaussian;
        gpu_ineff_avg.morphology += gpu_ineff_times.morphology;
        gpu_ineff_avg.canny += gpu_ineff_times.canny;
        gpu_ineff_avg.histogram += gpu_ineff_times.histogram;
        gpu_ineff_avg.total += gpu_ineff_times.total;
        gpu_ineff_avg.transfers = gpu_ineff_times.transfers;
        
        // GPU Optimizado
        gpu_opt_times = processGPU_Optimized(frame, gpu_optimized_result);
        gpu_opt_avg.gaussian += gpu_opt_times.gaussian;
        gpu_opt_avg.morphology += gpu_opt_times.morphology;
        gpu_opt_avg.canny += gpu_opt_times.canny;
        gpu_opt_avg.histogram += gpu_opt_times.histogram;
        gpu_opt_avg.total += gpu_opt_times.total;
        gpu_opt_avg.transfers = gpu_opt_times.transfers;
    }
    
    // Calcular promedios
    cpu_avg.gaussian /= iterations;
    cpu_avg.morphology /= iterations;
    cpu_avg.canny /= iterations;
    cpu_avg.histogram /= iterations;
    cpu_avg.total /= iterations;
    
    gpu_ineff_avg.gaussian /= iterations;
    gpu_ineff_avg.morphology /= iterations;
    gpu_ineff_avg.canny /= iterations;
    gpu_ineff_avg.histogram /= iterations;
    gpu_ineff_avg.total /= iterations;
    
    gpu_opt_avg.gaussian /= iterations;
    gpu_opt_avg.morphology /= iterations;
    gpu_opt_avg.canny /= iterations;
    gpu_opt_avg.histogram /= iterations;
    gpu_opt_avg.total /= iterations;
    
    // Imprimir resultados
    printResults("CPU", cpu_avg);
    printResults("GPU (Pipeline CPU↔GPU - Ineficiente)", gpu_ineff_avg);
    printResults("GPU (Pipeline GPU-only - Optimizado)", gpu_opt_avg);
    
    // Análisis comparativo
    cout << "\n====================================\n";
    cout << "  ANALISIS COMPARATIVO\n";
    cout << "====================================\n";
    double speedup_opt = cpu_avg.total / gpu_opt_avg.total;
    double speedup_ineff = cpu_avg.total / gpu_ineff_avg.total;
    
    cout << "Speedup GPU-optimizado vs CPU: " << speedup_opt << "x\n";
    cout << "Speedup GPU-ineficiente vs CPU: " << speedup_ineff << "x\n";
    cout << "Mejora GPU-optimizado vs GPU-ineficiente: " 
         << (gpu_ineff_avg.total / gpu_opt_avg.total) << "x\n";
    
    cout << "\nOverhead de transferencias CPU↔GPU:\n";
    cout << "Pipeline ineficiente: " << gpu_ineff_avg.transfers << " transferencias\n";
    cout << "Pipeline optimizado:  " << gpu_opt_avg.transfers << " transferencias\n";
    
    // Mostrar resultados visuales
    imshow("Original", frame);
    imshow("CPU Result", cpu_result);
    imshow("GPU Inefficient Result", gpu_inefficient_result);
    imshow("GPU Optimized Result", gpu_optimized_result);
    
    cout << "\nPresiona cualquier tecla para salir...\n";
    waitKey(0);
    
    return 0;
}