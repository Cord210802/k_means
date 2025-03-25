#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <omp.h>

// Estructura para representar un punto 2D
struct Point
{
    double x;
    double y;
    int cluster;

    Point(double x, double y) : x(x), y(y), cluster(-1) {}
};

// Estructura para representar un centroide
struct Centroid
{
    double x;
    double y;
    std::vector<int> points; // Índices de los puntos asignados a este centroide

    Centroid(double x, double y) : x(x), y(y) {}

    void clear()
    {
        points.clear();
    }
};

// Función para calcular la distancia euclidiana entre un punto y un centroide
double distance(const Point &point, const Centroid &centroid)
{
    return std::sqrt(std::pow(point.x - centroid.x, 2) + std::pow(point.y - centroid.y, 2));
}

// Implementación serial del algoritmo K-means
class KMeansSerial
{
private:
    int k;                           // Número de clusters
    std::vector<Point> points;       // Vector de puntos
    std::vector<Centroid> centroids; // Vector de centroides
    std::mt19937 rng;                // Generador de números aleatorios

public:
    KMeansSerial(int k) : k(k)
    {
        // Inicializar el generador de números aleatorios con una semilla fija para reproducibilidad
        rng = std::mt19937(42);
    }

    // Cargar los puntos desde un archivo CSV
    bool loadPoints(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
            return false;
        }

        points.clear();
        std::string line;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string x_str, y_str;

            if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ','))
            {
                try
                {
                    double x = std::stod(x_str);
                    double y = std::stod(y_str);
                    points.emplace_back(x, y);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error al convertir a número: " << e.what() << std::endl;
                }
            }
        }

        file.close();

        if (points.empty())
        {
            std::cerr << "Error: No se cargaron puntos desde el archivo." << std::endl;
            return false;
        }

        return true;
    }

    // Guardar los resultados en un archivo CSV
    bool saveResults(const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
            return false;
        }

        // Escribir cada punto con su cluster asignado
        for (const auto &point : points)
        {
            file << std::fixed << std::setprecision(3)
                 << point.x << ","
                 << point.y << ","
                 << point.cluster << std::endl;
        }

        file.close();
        return true;
    }

    // Paso 1: Inicializar los centroides aleatoriamente
    void initializeCentroids()
    {
        if (points.empty())
        {
            std::cerr << "Error: No hay puntos para inicializar centroides." << std::endl;
            return;
        }

        centroids.clear();

        // Crear una distribución uniforme real entre el mínimo y máximo de los puntos
        double min_x = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::min();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::min();

        for (const auto &point : points)
        {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }

        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);

        // Crear k centroides con posiciones aleatorias
        for (int i = 0; i < k; i++)
        {
            double x = dist_x(rng);
            double y = dist_y(rng);
            centroids.emplace_back(x, y);
        }
    }

    // Paso 2: Asignar cada punto al centroide más cercano
    bool assignPointsToCentroids()
    {
        bool changed = false;

        // Limpiar las asignaciones anteriores
        for (auto &centroid : centroids)
        {
            centroid.clear();
        }

        // Asignar cada punto al centroide más cercano
        for (size_t i = 0; i < points.size(); i++)
        {
            double min_distance = std::numeric_limits<double>::max();
            int closest_centroid = -1;

            for (size_t j = 0; j < centroids.size(); j++)
            {
                double dist = distance(points[i], centroids[j]);
                if (dist < min_distance)
                {
                    min_distance = dist;
                    closest_centroid = j;
                }
            }

            if (closest_centroid != -1)
            {
                // Verificar si la asignación del cluster cambió
                if (points[i].cluster != closest_centroid)
                {
                    changed = true;
                    points[i].cluster = closest_centroid;
                }
                centroids[closest_centroid].points.push_back(i);
            }
        }

        return changed;
    }

    // Paso 3: Actualizar la posición de los centroides
    void updateCentroids()
    {
        for (size_t i = 0; i < centroids.size(); i++)
        {
            if (centroids[i].points.empty())
            {
                continue; // Si no hay puntos asignados, no actualizar
            }

            double sum_x = 0.0;
            double sum_y = 0.0;

            // Calcular el promedio de las posiciones de los puntos asignados
            for (size_t j = 0; j < centroids[i].points.size(); j++)
            {
                int point_idx = centroids[i].points[j];
                sum_x += points[point_idx].x;
                sum_y += points[point_idx].y;
            }

            // Actualizar la posición del centroide
            centroids[i].x = sum_x / centroids[i].points.size();
            centroids[i].y = sum_y / centroids[i].points.size();
        }
    }

    // Ejecutar el algoritmo K-means
    void run(int max_iterations = 100)
    {
        if (points.empty() || k <= 0)
        {
            std::cerr << "Error: No hay puntos o el número de clusters es inválido." << std::endl;
            return;
        }

        // Paso 1: Inicializar centroides
        initializeCentroids();

        bool changed = true;
        int iteration = 0;

        // Iterar hasta que no haya cambios o se alcance el número máximo de iteraciones
        while (changed && iteration < max_iterations)
        {
            // Paso 2: Asignar puntos a centroides
            changed = assignPointsToCentroids();

            // Paso 3: Actualizar posición de centroides
            updateCentroids();

            iteration++;
        }
    }
};

// Implementación paralela del algoritmo K-means con OpenMP
class KMeansParallel
{
private:
    int k;                           // Número de clusters
    std::vector<Point> points;       // Vector de puntos
    std::vector<Centroid> centroids; // Vector de centroides
    std::mt19937 rng;                // Generador de números aleatorios
    int num_threads;                 // Número de hilos para OpenMP

public:
    KMeansParallel(int k, int threads = 4) : k(k), num_threads(threads)
    {
        // Inicializar el generador de números aleatorios con una semilla fija para reproducibilidad
        rng = std::mt19937(42);

        // Establecer el número de hilos para OpenMP
        omp_set_num_threads(num_threads);

        // Habilitar paralelismo anidado si es necesario
        omp_set_nested(1);
    }

    // Cargar los puntos desde un archivo CSV
    bool loadPoints(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
            return false;
        }

        points.clear();
        std::string line;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string x_str, y_str;

            if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ','))
            {
                try
                {
                    double x = std::stod(x_str);
                    double y = std::stod(y_str);
                    points.emplace_back(x, y);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error al convertir a número: " << e.what() << std::endl;
                }
            }
        }

        file.close();

        if (points.empty())
        {
            std::cerr << "Error: No se cargaron puntos desde el archivo." << std::endl;
            return false;
        }

        return true;
    }

    // Guardar los resultados en un archivo CSV
    bool saveResults(const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
            return false;
        }

        // Escribir cada punto con su cluster asignado
        for (const auto &point : points)
        {
            file << std::fixed << std::setprecision(3)
                 << point.x << ","
                 << point.y << ","
                 << point.cluster << std::endl;
        }

        file.close();
        return true;
    }

    // Paso 1: Inicializar los centroides aleatoriamente
    void initializeCentroids()
    {
        if (points.empty())
        {
            std::cerr << "Error: No hay puntos para inicializar centroides." << std::endl;
            return;
        }

        centroids.clear();

        // Optimizamos la búsqueda de mínimos y máximos con reduction
        double min_x = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::lowest();

#pragma omp parallel for reduction(min : min_x, min_y) reduction(max : max_x, max_y)
        for (size_t i = 0; i < points.size(); i++)
        {
            min_x = std::min(min_x, points[i].x);
            max_x = std::max(max_x, points[i].x);
            min_y = std::min(min_y, points[i].y);
            max_y = std::max(max_y, points[i].y);
        }

        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);

        // Crear k centroides con posiciones aleatorias
        for (int i = 0; i < k; i++)
        {
            double x = dist_x(rng);
            double y = dist_y(rng);
            centroids.emplace_back(x, y);
        }
    }

    // Paso 2: Asignar cada punto al centroide más cercano
    bool assignPointsToCentroids()
    {
        bool changed = false;

        // Limpiar las asignaciones anteriores
        for (auto &centroid : centroids)
        {
            centroid.clear();
        }

// Paralelizar la asignación de puntos con reducción booleana para 'changed'
#pragma omp parallel reduction(|| : changed)
        {
            // Creamos estructuras locales para cada hilo
            std::vector<std::vector<int>> local_centroid_points(centroids.size());

#pragma omp for
            for (size_t i = 0; i < points.size(); i++)
            {
                double min_distance = std::numeric_limits<double>::max();
                int closest_centroid = -1;

                for (size_t j = 0; j < centroids.size(); j++)
                {
                    double dist = distance(points[i], centroids[j]);
                    if (dist < min_distance)
                    {
                        min_distance = dist;
                        closest_centroid = j;
                    }
                }

                if (closest_centroid != -1)
                {
                    // Verificar si la asignación del cluster cambió
                    if (points[i].cluster != closest_centroid)
                    {
                        changed = true;
                        points[i].cluster = closest_centroid;
                    }
                    // Almacenar el punto localmente para este hilo
                    local_centroid_points[closest_centroid].push_back(i);
                }
            }

            // Ahora combinamos los resultados locales con una sección crítica
            for (size_t j = 0; j < centroids.size(); j++)
            {
                if (!local_centroid_points[j].empty())
                {
#pragma omp critical
                    {
                        centroids[j].points.insert(
                            centroids[j].points.end(),
                            local_centroid_points[j].begin(),
                            local_centroid_points[j].end());
                    }
                }
            }
        }

        return changed;
    }

    // Paso 3: Actualizar la posición de los centroides
    void updateCentroids()
    {
#pragma omp parallel for
        for (size_t i = 0; i < centroids.size(); i++)
        {
            if (centroids[i].points.empty())
            {
                continue; // Si no hay puntos asignados, no actualizar
            }

            double sum_x = 0.0;
            double sum_y = 0.0;
            size_t count = centroids[i].points.size();

            // Calculamos la suma secuencialmente para evitar paralelismo anidado
            for (size_t j = 0; j < count; j++)
            {
                int point_idx = centroids[i].points[j];
                sum_x += points[point_idx].x;
                sum_y += points[point_idx].y;
            }

            // Actualizar la posición del centroide
            centroids[i].x = sum_x / count;
            centroids[i].y = sum_y / count;
        }
    }

    // Ejecutar el algoritmo K-means
    void run(int max_iterations = 100)
    {
        if (points.empty() || k <= 0)
        {
            std::cerr << "Error: No hay puntos o el número de clusters es inválido." << std::endl;
            return;
        }

        // Paso 1: Inicializar centroides
        initializeCentroids();

        bool changed = true;
        int iteration = 0;

        // Iterar hasta que no haya cambios o se alcance el número máximo de iteraciones
        while (changed && iteration < max_iterations)
        {
            // Paso 2: Asignar puntos a centroides
            changed = assignPointsToCentroids();

            if (changed)
            {
                updateCentroids();
            }
            iteration++;
        }
    }

    // Método para establecer el número de hilos
    void setNumThreads(int threads)
    {
        num_threads = threads;
        omp_set_num_threads(num_threads);
    }

    // Método para obtener el número de hilos
    int getNumThreads() const
    {
        return num_threads;
    }
};

// Estructura para almacenar los resultados del benchmark
struct BenchmarkResult
{
    int num_points;
    int num_threads;
    double serial_time;
    double parallel_time;
    double speedup;
};

// Función para ejecutar un benchmark para una combinación específica de puntos e hilos
BenchmarkResult runBenchmark(const std::string &data_file, int k, int num_threads, int iterations = 10)
{
    BenchmarkResult result;

    // Extraer el número de puntos del nombre del archivo
    std::string filename = data_file.substr(data_file.find_last_of("/\\") + 1);
    result.num_points = std::stoi(filename.substr(0, filename.find("_")));
    result.num_threads = num_threads;

    // Inicializar algoritmos
    KMeansSerial kmeans_serial(k);
    KMeansParallel kmeans_parallel(k, num_threads);

    // Cargar datos solo una vez para evitar afectar los tiempos
    if (!kmeans_serial.loadPoints(data_file) || !kmeans_parallel.loadPoints(data_file))
    {
        std::cerr << "Error al cargar datos desde " << data_file << std::endl;
        return result;
    }

    // Ejecutar versión serial varias veces y promediar
    double total_serial_time = 0.0;
    for (int i = 0; i < iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        kmeans_serial.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_serial_time += elapsed.count();
    }
    result.serial_time = total_serial_time / iterations;

    // Ejecutar versión paralela varias veces y promediar
    double total_parallel_time = 0.0;
    for (int i = 0; i < iterations; i++)
    {
        double start_time = omp_get_wtime();
        kmeans_parallel.run();
        double end_time = omp_get_wtime();
        double elapsed_time = (end_time - start_time) * 1000.0; // Convertir a milisegundos
        total_parallel_time += elapsed_time;
    }
    result.parallel_time = total_parallel_time / iterations;

    // Calcular speedup
    result.speedup = result.serial_time / result.parallel_time;

    return result;
}

int main()
{
    // Configuración del benchmark
    const int k = 5; // Número fijo de clusters
    const int max_iterations = 100;
    const int num_runs = 10; // Número de iteraciones para promediar

    // Vectores con los tamaños de puntos y números de hilos a probar
    std::vector<int> point_sizes = {100000, 200000, 300000, 400000, 600000, 800000, 1000000};

    // Obtener el número máximo de hilos disponibles (cores virtuales)
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {1, max_threads / 2, max_threads, max_threads * 2};

    std::cout << "Cores virtuales detectados: " << max_threads << std::endl;
    std::cout << "Configuraciones de hilos a probar: ";
    for (int threads : thread_counts)
    {
        std::cout << threads << " ";
    }
    std::cout << std::endl
              << std::endl;

    // Vector para almacenar todos los resultados
    std::vector<BenchmarkResult> all_results;

    // Ejecutar benchmarks para todas las combinaciones
    for (int points : point_sizes)
    {
        std::string data_file = "./" + std::to_string(points) + "_data.csv";

        // Verificar si el archivo existe
        std::ifstream file_check(data_file);
        if (!file_check)
        {
            std::cerr << "Archivo no encontrado: " << data_file << ". Saltando esta configuración." << std::endl;
            continue;
        }
        file_check.close();

        std::cout << "Procesando " << points << " puntos:" << std::endl;

        for (int threads : thread_counts)
        {
            std::cout << "  Con " << threads << " hilos... ";
            BenchmarkResult result = runBenchmark(data_file, k, threads, num_runs);
            all_results.push_back(result);

            std::cout << "Tiempo serial: " << std::fixed << std::setprecision(2) << result.serial_time
                      << " ms, Tiempo paralelo: " << result.parallel_time
                      << " ms, Speedup: " << std::setprecision(2) << result.speedup << "x" << std::endl;
        }
        std::cout << std::endl;
    }

    // Guardar todos los resultados en un archivo CSV
    std::ofstream results_file("benchmark_results.csv");
    if (results_file.is_open())
    {
        results_file << "Puntos,Hilos,Tiempo Serial (ms),Tiempo Paralelo (ms),Speedup\n";

        for (const auto &result : all_results)
        {
            results_file << result.num_points << ","
                         << result.num_threads << ","
                         << std::fixed << std::setprecision(2) << result.serial_time << ","
                         << result.parallel_time << ","
                         << std::setprecision(4) << result.speedup << "\n";
        }

        results_file.close();
        std::cout << "Resultados guardados en benchmark_results.csv" << std::endl;
    }

    // Mostrar tabla de resultados final
    std::cout << "\n=== RESUMEN DE RESULTADOS ===\n";
    std::cout << "| Puntos  | Hilos | Tiempo Serial (ms) | Tiempo Paralelo (ms) | Speedup |\n";
    std::cout << "|---------|-------|---------------------|----------------------|--------|\n";

    for (const auto &result : all_results)
    {
        std::cout << "| " << std::setw(7) << result.num_points
                  << " | " << std::setw(5) << result.num_threads
                  << " | " << std::setw(19) << std::fixed << std::setprecision(2) << result.serial_time
                  << " | " << std::setw(20) << result.parallel_time
                  << " | " << std::setw(6) << std::setprecision(2) << result.speedup << "x |\n";
    }

    return 0;
}