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
#include <chrono> // Para medir el tiempo

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

int main()
{
    // Ruta fija para el archivo de entrada
    std::string input_file = "./1000_data.csv";

    // Número fijo de clusters - MODIFICAR AQUÍ PARA CAMBIAR EL VALOR
    const int k = 5;

    // Número máximo de iteraciones
    const int max_iterations = 100;

    // Extraer el número de puntos del nombre del archivo (asumiendo formato "n_data.csv")
    std::string n_points_str = input_file.substr(0, input_file.find("_"));
    std::string output_file = n_points_str + "_results.csv";

    KMeansSerial kmeans(k);

    // Cargar los puntos
    if (!kmeans.loadPoints(input_file))
    {
        return 1;
    }

    // Medir el tiempo solo para el algoritmo K-means
    auto start = std::chrono::high_resolution_clock::now();

    // Ejecutar el algoritmo
    kmeans.run(max_iterations);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Imprimir solo el tiempo total
    std::cout << "Tiempo total del algoritmo K-means: " << elapsed.count() << " ms" << std::endl;

    // Guardar los resultados
    if (!kmeans.saveResults(output_file))
    {
        return 1;
    }

    return 0;
}