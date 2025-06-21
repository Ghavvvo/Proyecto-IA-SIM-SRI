"""
Script de ejemplo para demostrar el uso de GLiNER con el crawler
"""
from crawler import TourismCrawler

def test_gliner_crawler():
    """
    Ejemplo de uso del crawler con GLiNER para procesar información turística
    """
    # URLs de ejemplo para crawling
    starting_urls = [
        "https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html"
    ]
    
    # Crear instancia del crawler con Gemini deshabilitado por defecto
    crawler = TourismCrawler(
        starting_urls=starting_urls,
        chroma_collection_name="tourism_gliner_test",
        max_pages=10,
        max_depth=2,
        num_threads=3,
        enable_gemini_processing=False  # Deshabilitamos Gemini
    )
    
    # Habilitar GLiNER
    print("\n🔄 Habilitando procesamiento con GLiNER...")
    if crawler.enable_gliner():
        print("✅ GLiNER habilitado exitosamente")
    else:
        print("❌ No se pudo habilitar GLiNER")
        return
    
    # Ejecutar crawler
    print("\n🚀 Iniciando crawler con GLiNER...")
    pages_added = crawler.run_parallel_crawler()
    
    print(f"\n✅ Proceso completado. {pages_added} páginas añadidas a la base de datos.")
    
    # Mostrar algunas estadísticas
    print("\n📊 Estadísticas de procesamiento:")
    print(f"   • Páginas procesadas con GLiNER: {crawler.gliner_processed}")
    print(f"   • Errores de GLiNER: {crawler.gliner_errors}")
    if crawler.gliner_agent:
        stats = crawler.gliner_agent.get_stats()
        print(f"   • Tasa de éxito: {stats['success_rate']:.2%}")


def test_gliner_with_keywords():
    """
    Ejemplo de uso del crawler con GLiNER usando búsqueda por palabras clave
    """
    # Palabras clave para buscar
    keywords = ["hoteles cuba", "varadero beach resort"]
    
    # Crear instancia del crawler
    crawler = TourismCrawler(
        starting_urls=[],  # No necesitamos URLs iniciales
        chroma_collection_name="tourism_gliner_keywords",
        max_pages=15,
        max_depth=1,
        num_threads=5,
        enable_gemini_processing=False
    )
    
    # Habilitar GLiNER
    print("\n🔄 Habilitando procesamiento con GLiNER...")
    if not crawler.enable_gliner():
        print("❌ No se pudo habilitar GLiNER")
        return
    
    # Ejecutar crawler con palabras clave
    print(f"\n🔍 Buscando información sobre: {keywords}")
    pages_added = crawler.run_parallel_crawler_from_keywords(
        keywords=keywords,
        max_depth=1
    )
    
    print(f"\n✅ Proceso completado. {pages_added} páginas añadidas a la base de datos.")


def compare_processors():
    """
    Compara el procesamiento entre Gemini y GLiNER
    """
    test_url = ["https://www.lonelyplanet.com/cuba"]
    
    print("\n=== COMPARACIÓN DE PROCESADORES ===\n")
    
    # Test con Gemini
    print("1️⃣ Procesando con Gemini...")
    crawler_gemini = TourismCrawler(
        starting_urls=test_url,
        chroma_collection_name="tourism_comparison_gemini",
        max_pages=5,
        max_depth=1,
        num_threads=2,
        enable_gemini_processing=True
    )
    pages_gemini = crawler_gemini.run_parallel_crawler()
    
    # Test con GLiNER
    print("\n2️⃣ Procesando con GLiNER...")
    crawler_gliner = TourismCrawler(
        starting_urls=test_url,
        chroma_collection_name="tourism_comparison_gliner",
        max_pages=5,
        max_depth=1,
        num_threads=2,
        enable_gemini_processing=False
    )
    crawler_gliner.enable_gliner()
    pages_gliner = crawler_gliner.run_parallel_crawler()
    
    # Comparar resultados
    print("\n📊 RESULTADOS DE LA COMPARACIÓN:")
    print(f"\nGemini:")
    print(f"  • Páginas procesadas: {crawler_gemini.gemini_processed}")
    print(f"  • Errores: {crawler_gemini.gemini_errors}")
    
    print(f"\nGLiNER:")
    print(f"  • Páginas procesadas: {crawler_gliner.gliner_processed}")
    print(f"  • Errores: {crawler_gliner.gliner_errors}")


if __name__ == "__main__":
    print("🎯 Ejemplos de uso de GLiNER con el crawler de turismo\n")
    print("Seleccione una opción:")
    print("1. Test básico con URLs predefinidas")
    print("2. Test con búsqueda por palabras clave")
    print("3. Comparación entre Gemini y GLiNER")
    
    choice = input("\nIngrese su opción (1-3): ")
    
    if choice == "1":
        test_gliner_crawler()
    elif choice == "2":
        test_gliner_with_keywords()
    elif choice == "3":
        compare_processors()
    else:
        print("Opción no válida")