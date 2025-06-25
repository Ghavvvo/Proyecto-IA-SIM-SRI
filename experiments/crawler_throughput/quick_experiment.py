#!/usr/bin/env python3
"""
Script de experimentación rápida para el SimplifiedCrawler.
Versión simplificada con menos configuraciones para pruebas rápidas.
"""

import sys
import os
import csv
import time
import psutil
from datetime import datetime
from typing import List, Dict

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from simplified_crawler import SimplifiedCrawler


def run_quick_experiment():
    """Ejecuta un experimento rápido con configuraciones básicas"""

    print("🚀 EXPERIMENTO RÁPIDO DE THROUGHPUT")
    print("=" * 40)

    # URLs de prueba
    test_urls = [
        "https://www.tripadvisor.com/",
        "https://www.booking.com/",
        "https://www.expedia.com/",
        "https://www.airbnb.com/",
        "https://www.hotels.com/",
        "https://www.kayak.com/",
        "https://www.priceline.com/",
        "https://www.agoda.com/",
        "https://www.orbitz.com/",
        "https://www.trivago.com/"
    ]

    # Configuraciones simplificadas
    configs = [
        {"threads": 1, "pages": 50, "depth": 1},
        {"threads": 5, "pages": 50, "depth": 1},
        {"threads": 10, "pages": 50, "depth": 1},
        {"threads": 5, "pages": 50, "depth": 2},
        {"threads": 10, "pages": 50, "depth": 2},
    ]

    results = []
    csv_filename = f"quick_throughput_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

    # Preparar CSV con campos ampliados
    fieldnames = [
        "experiment_id", "timestamp", "num_threads", "max_pages", "max_depth",
        "pages_processed", "pages_extracted", "urls_discovered", "total_time", "pages_per_second",
        "extraction_rate", "error_rate", "thread_efficiency",
        "avg_process_time", "avg_request_time", "avg_extraction_time",
        "p50_process_time", "p95_process_time", "p99_process_time",
        "cpu_before", "cpu_after", "memory_before", "memory_after", "memory_available_mb",
        "success"
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, config in enumerate(configs, 1):
            print(f"\n🧪 Experimento {i}/{len(configs)}")
            print(f"   Configuración: {config}")

            try:
                # Medir CPU y memoria antes de iniciar
                process = psutil.Process()
                cpu_before = psutil.cpu_percent(interval=0.1)
                memory_before = process.memory_info().rss / (1024 * 1024)  # MB
                memory_available_before = psutil.virtual_memory().available / (1024 * 1024)  # MB

                # Crear y ejecutar crawler
                crawler = SimplifiedCrawler(
                    starting_urls=test_urls,
                    max_pages=config["pages"],
                    max_depth=config["depth"],
                    num_threads=config["threads"]
                )

                start_time = time.time()
                metrics = crawler.run_crawler()

                # Medir CPU y memoria después
                cpu_after = psutil.cpu_percent(interval=0.1)
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_available_after = psutil.virtual_memory().available / (1024 * 1024)  # MB

                # Preparar resultado con métricas extendidas
                result = {
                    "experiment_id": i,
                    "timestamp": datetime.now().isoformat(),
                    "num_threads": config["threads"],
                    "max_pages": config["pages"],
                    "max_depth": config["depth"],
                    "pages_processed": metrics["results"]["pages_processed"],
                    "pages_extracted": metrics["results"]["pages_extracted"],
                    "urls_discovered": metrics["results"]["urls_discovered"],
                    "total_time": metrics["performance"]["total_time"],
                    "pages_per_second": metrics["performance"]["pages_per_second"],
                    "extraction_rate": metrics["efficiency"]["extraction_rate"],
                    "error_rate": metrics["efficiency"]["error_rate"],
                    "thread_efficiency": metrics["efficiency"]["thread_efficiency"],
                    "avg_process_time": metrics["performance"]["average_process_time"],
                    "avg_request_time": metrics["performance"]["average_request_time"],
                    "avg_extraction_time": metrics["performance"]["average_extraction_time"],
                    "p50_process_time": metrics["performance"].get("p50_process_time", 0),
                    "p95_process_time": metrics["performance"].get("p95_process_time", 0),
                    "p99_process_time": metrics["performance"].get("p99_process_time", 0),
                    "cpu_before": cpu_before,
                    "cpu_after": cpu_after,
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "memory_available_mb": memory_available_after,
                    "success": True
                }

                # Guardar resultado
                writer.writerow(result)
                results.append(result)

                print(f"   ✅ Completado: {result['pages_per_second']:.2f} páginas/seg")
                print(f"   🧠 Memoria: {result['memory_after'] - result['memory_before']:.1f}MB adicionales")
                print(f"   ⏱️ Tiempo promedio por página: {result['avg_process_time']*1000:.1f}ms")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

            # Pausa breve entre experimentos
            if i < len(configs):
                time.sleep(2)

    # Resumen
    print(f"\n📊 RESUMEN DE RESULTADOS")
    print(f"📄 Archivo: {csv_filename}")

    if results:
        best = max(results, key=lambda x: x["pages_per_second"])
        print(f"🏆 Mejor throughput: {best['pages_per_second']:.2f} páginas/seg")
        print(f"   └─ {best['num_threads']} threads, {best['max_pages']} páginas")

        most_efficient = max(results, key=lambda x: x["thread_efficiency"])
        print(f"⚡ Mejor eficiencia por hilo: {most_efficient['thread_efficiency']:.3f}")
        print(f"   └─ {most_efficient['num_threads']} threads")

        fastest_processing = min(results, key=lambda x: x["avg_process_time"])
        print(f"⏱️ Tiempo de procesamiento más rápido: {fastest_processing['avg_process_time']*1000:.1f}ms")
        print(f"   └─ {fastest_processing['num_threads']} threads")

        print(f"\n📈 Todos los resultados:")
        for r in results:
            print(f"   {r['num_threads']:2d} threads: {r['pages_per_second']:6.2f} pág/s | "
                  f"CPU: {r['cpu_after']-r['cpu_before']:+.1f}% | "
                  f"Mem: {r['memory_after']-r['memory_before']:+.1f}MB | "
                  f"Extr: {r['extraction_rate']:.1%}")


if __name__ == "__main__":
    run_quick_experiment()
