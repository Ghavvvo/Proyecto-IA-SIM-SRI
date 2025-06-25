#!/usr/bin/env python3
"""
Script para analizar los resultados de los experimentos de throughput.
"""

import os
import csv
import glob
from datetime import datetime
from typing import List, Dict
import statistics


class ThroughputAnalyzer:
    """Analizador de resultados de experimentos de throughput"""
    
    def __init__(self, csv_file_path: str = None):
        self.csv_file_path = csv_file_path
        self.results = []
        
    def find_latest_csv(self) -> str:
        """Encuentra el archivo CSV más reciente"""
        csv_pattern = os.path.join(os.path.dirname(__file__), "*throughput*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise FileNotFoundError("No se encontraron archivos CSV de resultados")
        
        # Ordenar por fecha de modificación
        latest_file = max(csv_files, key=os.path.getmtime)
        return latest_file
    
    def load_results(self):
        """Carga los resultados desde el archivo CSV"""
        if not self.csv_file_path:
            self.csv_file_path = self.find_latest_csv()
        
        print(f"📄 Cargando resultados desde: {os.path.basename(self.csv_file_path)}")
        
        with open(self.csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            self.results = []
            
            for row in reader:
                # Convertir valores numéricos
                for key in row:
                    if key in ['experiment_id', 'num_threads', 'max_pages', 'max_depth', 
                              'starting_urls_count', 'pages_processed', 'pages_extracted', 
                              'errors_count', 'urls_discovered', 'urls_visited']:
                        try:
                            row[key] = int(row[key])
                        except (ValueError, TypeError):
                            row[key] = 0
                    elif key in ['total_time', 'pages_per_second', 'avg_process_time',
                               'avg_request_time', 'avg_extraction_time', 'extraction_rate',
                               'error_rate', 'thread_efficiency', 'p50_process_time',
                               'p95_process_time', 'p99_process_time', 'cpu_before',
                               'cpu_after', 'memory_before', 'memory_after', 'memory_available_mb']:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            row[key] = 0.0
                    elif key == 'success':
                        row[key] = row[key].lower() == 'true'
                
                self.results.append(row)
        
        print(f"✅ Cargados {len(self.results)} resultados")
    
    def filter_successful(self) -> List[Dict]:
        """Filtra solo los experimentos exitosos"""
        # Si los resultados provienen de quick_experiment, no tendrán
        # el campo "success" pero todos son considerados exitosos
        if self.results and 'success' not in self.results[0]:
            return self.results

        # Filtrado normal para resultados de throughput_experiment
        return [r for r in self.results if r.get('success', False)]
    
    def analyze_by_threads(self):
        """Analiza resultados por número de hilos"""
        successful = self.filter_successful()
        if not successful:
            print("❌ No hay resultados exitosos para analizar")
            return
        
        print(f"\n🧵 ANÁLISIS POR NÚMERO DE HILOS")
        print("=" * 50)
        
        # Agrupar por número de hilos
        by_threads = {}
        for result in successful:
            threads = result['num_threads']
            if threads not in by_threads:
                by_threads[threads] = []
            by_threads[threads].append(result)
        
        # Analizar cada grupo
        for threads in sorted(by_threads.keys()):
            results = by_threads[threads]
            
            throughputs = [r['pages_per_second'] for r in results]
            extraction_rates = [r['extraction_rate'] for r in results]
            thread_efficiencies = [r['thread_efficiency'] for r in results]
            
            print(f"\n📊 {threads:2d} hilos ({len(results)} experimentos):")
            print(f"   Throughput promedio: {statistics.mean(throughputs):6.2f} páginas/seg")
            if len(throughputs) > 1:
                print(f"   Desviación estándar:  {statistics.stdev(throughputs):6.2f}")
            print(f"   Mejor throughput:     {max(throughputs):6.2f} páginas/seg")
            print(f"   Extracción promedio:  {statistics.mean(extraction_rates):6.1%}")
            print(f"   Eficiencia por hilo:  {statistics.mean(thread_efficiencies):6.3f}")
    
    def analyze_by_pages(self):
        """Analiza resultados por número de páginas"""
        successful = self.filter_successful()
        if not successful:
            return
        
        print(f"\n📄 ANÁLISIS POR NÚMERO DE PÁGINAS")
        print("=" * 50)
        
        by_pages = {}
        for result in successful:
            pages = result['max_pages']
            if pages not in by_pages:
                by_pages[pages] = []
            by_pages[pages].append(result)
        
        for pages in sorted(by_pages.keys()):
            results = by_pages[pages]
            throughputs = [r['pages_per_second'] for r in results]
            
            print(f"\n📊 {pages:3d} páginas ({len(results)} experimentos):")
            print(f"   Throughput promedio: {statistics.mean(throughputs):6.2f} páginas/seg")
            print(f"   Mejor throughput:    {max(throughputs):6.2f} páginas/seg")
    
    def analyze_performance_metrics(self):
        """Analiza métricas de rendimiento detalladas"""
        successful = self.filter_successful()
        if not successful:
            return

        print(f"\n⚡ MÉTRICAS DE RENDIMIENTO DETALLADAS")
        print("=" * 50)

        # Analizar tiempos de procesamiento
        if all(r.get('avg_process_time') for r in successful):
            process_times = [r['avg_process_time'] for r in successful]
            request_times = [r['avg_request_time'] for r in successful if 'avg_request_time' in r]
            extraction_times = [r['avg_extraction_time'] for r in successful if 'avg_extraction_time' in r]

            print(f"\n⏱️ Tiempos de procesamiento (segundos):")
            print(f"   Promedio general:     {statistics.mean(process_times):.4f}s")
            print(f"   Mediana:              {statistics.median(process_times):.4f}s")
            print(f"   Mínimo:               {min(process_times):.4f}s")
            print(f"   Máximo:               {max(process_times):.4f}s")

            if request_times:
                print(f"\n   Tiempo de solicitud:   {statistics.mean(request_times):.4f}s ({statistics.mean(request_times)/statistics.mean(process_times)*100:.1f}% del total)")

            if extraction_times:
                print(f"   Tiempo de extracción: {statistics.mean(extraction_times):.4f}s ({statistics.mean(extraction_times)/statistics.mean(process_times)*100:.1f}% del total)")

        # Analizar uso de CPU y memoria
        if all(r.get('cpu_before') is not None for r in successful):
            cpu_diffs = [r['cpu_after'] - r['cpu_before'] for r in successful]
            mem_diffs = [r['memory_after'] - r['memory_before'] for r in successful]

            print(f"\n🖥️ Uso de recursos:")
            print(f"   Incremento CPU promedio:    {statistics.mean(cpu_diffs):+.2f}%")
            print(f"   Incremento memoria promedio:{statistics.mean(mem_diffs):+.2f} MB")

            # Analizar correlación entre threads y uso de recursos
            by_threads = {}
            for result in successful:
                threads = result['num_threads']
                if threads not in by_threads:
                    by_threads[threads] = []
                by_threads[threads].append(result)

            print(f"\n📊 Consumo de recursos por número de threads:")
            for threads in sorted(by_threads.keys()):
                results = by_threads[threads]
                cpu_use = [r['cpu_after'] - r['cpu_before'] for r in results]
                mem_use = [r['memory_after'] - r['memory_before'] for r in results]

                print(f"   {threads:2d} threads: CPU: {statistics.mean(cpu_use):+6.2f}% | "
                      f"Memoria: {statistics.mean(mem_use):+6.2f} MB")

    def find_optimal_configurations(self):
        """Encuentra las configuraciones óptimas"""
        successful = self.filter_successful()
        if not successful:
            return
        
        print(f"\n🏆 CONFIGURACIONES ÓPTIMAS")
        print("=" * 50)
        
        # Mejor throughput absoluto
        best_throughput = max(successful, key=lambda x: x['pages_per_second'])
        print(f"\n🚀 Mayor throughput: {best_throughput['pages_per_second']:.2f} páginas/seg")
        print(f"   Configuración: {best_throughput['num_threads']} hilos, "
              f"{best_throughput['max_pages']} páginas, profundidad {best_throughput['max_depth']}")
        
        # Mejor eficiencia por hilo
        best_efficiency = max(successful, key=lambda x: x['thread_efficiency'])
        print(f"\n⚡ Mayor eficiencia por hilo: {best_efficiency['thread_efficiency']:.3f}")
        print(f"   Configuración: {best_efficiency['num_threads']} hilos, "
              f"{best_efficiency['max_pages']} páginas, profundidad {best_efficiency['max_depth']}")
        
        # Mejor tasa de extracción
        best_extraction = max(successful, key=lambda x: x['extraction_rate'])
        print(f"\n📈 Mayor tasa de extracción: {best_extraction['extraction_rate']:.1%}")
        print(f"   Configuración: {best_extraction['num_threads']} hilos, "
              f"{best_extraction['max_pages']} páginas, profundidad {best_extraction['max_depth']}")
        
        # Menor tasa de errores
        best_reliability = min(successful, key=lambda x: x['error_rate'])
        print(f"\n🛡️ Menor tasa de errores: {best_reliability['error_rate']:.1%}")
        print(f"   Configuración: {best_reliability['num_threads']} hilos, "
              f"{best_reliability['max_pages']} páginas, profundidad {best_reliability['max_depth']}")
    
    def generate_summary_stats(self):
        """Genera estadísticas generales"""
        successful = self.filter_successful()
        if not successful:
            return
        
        print(f"\n📊 ESTADÍSTICAS GENERALES")
        print("=" * 50)
        
        total_experiments = len(self.results)
        successful_count = len(successful)
        
        print(f"Total de experimentos:     {total_experiments}")
        print(f"Experimentos exitosos:     {successful_count}")
        print(f"Tasa de éxito:            {successful_count/total_experiments:.1%}")
        
        if successful:
            throughputs = [r['pages_per_second'] for r in successful]
            extraction_rates = [r['extraction_rate'] for r in successful]
            error_rates = [r['error_rate'] for r in successful]
            
            print(f"\nThroughput:")
            print(f"   Promedio:  {statistics.mean(throughputs):6.2f} páginas/seg")
            print(f"   Mediana:   {statistics.median(throughputs):6.2f} páginas/seg")
            print(f"   Máximo:    {max(throughputs):6.2f} páginas/seg")
            print(f"   Mínimo:    {min(throughputs):6.2f} páginas/seg")
            
            print(f"\nTasa de extracción:")
            print(f"   Promedio:  {statistics.mean(extraction_rates):6.1%}")
            print(f"   Mediana:   {statistics.median(extraction_rates):6.1%}")
            
            print(f"\nTasa de errores:")
            print(f"   Promedio:  {statistics.mean(error_rates):6.1%}")
            print(f"   Mediana:   {statistics.median(error_rates):6.1%}")
    
    def export_summary(self):
        """Exporta un resumen a archivo de texto"""
        summary_file = self.csv_file_path.replace('.csv', '_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RESUMEN DE EXPERIMENTOS DE THROUGHPUT\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo fuente: {os.path.basename(self.csv_file_path)}\n")
            f.write("=" * 60 + "\n\n")
            
            successful = self.filter_successful()
            if successful:
                best_throughput = max(successful, key=lambda x: x['pages_per_second'])
                best_efficiency = max(successful, key=lambda x: x['thread_efficiency'])
                
                f.write(f"CONFIGURACIONES RECOMENDADAS:\n\n")
                f.write(f"Para máximo throughput:\n")
                f.write(f"  - Hilos: {best_throughput['num_threads']}\n")
                f.write(f"  - Páginas: {best_throughput['max_pages']}\n")
                f.write(f"  - Profundidad: {best_throughput['max_depth']}\n")
                f.write(f"  - Resultado: {best_throughput['pages_per_second']:.2f} páginas/seg\n\n")
                
                f.write(f"Para máxima eficiencia:\n")
                f.write(f"  - Hilos: {best_efficiency['num_threads']}\n")
                f.write(f"  - Páginas: {best_efficiency['max_pages']}\n")
                f.write(f"  - Profundidad: {best_efficiency['max_depth']}\n")
                f.write(f"  - Resultado: {best_efficiency['thread_efficiency']:.3f} eficiencia/hilo\n")
        
        print(f"\n📄 Resumen exportado a: {os.path.basename(summary_file)}")
    
    def run_analysis(self):
        """Ejecuta el análisis completo"""
        print("🔍 ANÁLISIS DE RESULTADOS DE THROUGHPUT")
        print("=" * 50)
        
        self.load_results()
        self.generate_summary_stats()
        self.analyze_by_threads()
        self.analyze_by_pages()
        self.analyze_performance_metrics()
        self.find_optimal_configurations()
        self.export_summary()


def main():
    """Función principal"""
    import sys
    
    csv_file = None
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        analyzer = ThroughputAnalyzer(csv_file)
        analyzer.run_analysis()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Ejecuta primero throughput_experiment.py para generar datos")
    
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

