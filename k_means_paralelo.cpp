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
#include <omp.h> // Incluir OpenMP

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

// Función inline para calcular la distancia euclidiana entre un punto y un centroide
inline double distance(const Point &point, const Centroid &centroid)
{
    return std::sqrt(std::pow(point.x - centroid.x, 2) + std::pow(point.y - centroid.y, 2));
}

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

        // Usamos un mutex para proteger las actualizaciones a centroids.points
        std::vector<omp_lock_t> centroid_locks(centroids.size());
        for (size_t i = 0; i < centroids.size(); i++)
        {
            omp_init_lock(&centroid_locks[i]);
        }

// Paralelizar la asignación de puntos con reducción booleana para 'changed'
#pragma omp parallel reduction(|| : changed)
        {
            int actual_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

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

            // Ahora combinamos los resultados locales con protección de locks
            for (size_t j = 0; j < centroids.size(); j++)
            {
                if (!local_centroid_points[j].empty())
                {
                    omp_set_lock(&centroid_locks[j]);
                    centroids[j].points.insert(
                        centroids[j].points.end(),
                        local_centroid_points[j].begin(),
                        local_centroid_points[j].end());
                    omp_unset_lock(&centroid_locks[j]);
                }
            }
        }

        // Liberamos los locks
        for (size_t i = 0; i < centroids.size(); i++)
        {
            omp_destroy_lock(&centroid_locks[i]);
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

        std::cout << "Convergencia alcanzada en " << iteration << " iteraciones." << std::endl;
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

int main()
{
    // Ruta fija para el archivo de entrada
    std::string input_file = "/home/cord2108/ITAM/Paralelo/proyecto/100000_data.csv";

    // Número fijo de clusters - MODIFICAR AQUÍ PARA CAMBIAR EL VALOR
    const int k = 5;

    // Número máximo de iteraciones
    const int max_iterations = 100;

    // Número de hilos para OpenMP (se puede ajustar según el sistema)
    const int num_threads = 4;

    // Extraer el número de puntos del nombre del archivo (asumiendo formato "n_data.csv")
    std::string n_points_str = input_file.substr(input_file.find_last_of("/") + 1);
    n_points_str = n_points_str.substr(0, n_points_str.find("_"));
    std::string output_file = n_points_str + "_results_op.csv";

    KMeansParallel kmeans(k, num_threads);

    std::cout << "Ejecutando K-means paralelo con " << kmeans.getNumThreads() << " hilos" << std::endl;

    // Cargar los puntos
    if (!kmeans.loadPoints(input_file))
    {
        return 1;
    }

    // Medir el tiempo usando las funciones de OpenMP
    double start_time = omp_get_wtime();

    // Ejecutar el algoritmo
    kmeans.run(max_iterations);

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time) * 1000.0; // Convertir a milisegundos

    // Imprimir solo el tiempo total
    std::cout << "Tiempo total del algoritmo K-means paralelo: " << elapsed_time << " ms" << std::endl;

    // Guardar los resultados
    if (!kmeans.saveResults(output_file))
    {
        return 1;
    }

    return 0;
}