from crawler import TourismCrawler
from rag import RAGSystem

# Ejemplo de uso
if __name__ == "__main__":
    starting_urls = [
        # Sitios de guías turísticas
        "https://www.lonelyplanet.com/destinations",
        "https://www.tripadvisor.com/Tourism",
        "https://www.roughguides.com/destinations/",
        "https://www.frommers.com/destinations",
        "https://www.nationalgeographic.com/travel/destinations/",

        # Blogs de viajes
        "https://www.nomadicmatt.com/travel-blog/",
        "https://www.bemytravelmuse.com/",
        "https://www.tripsavvy.com/",

        # Sitios de reseñas de hoteles y destinos
        "https://www.booking.com/reviews.html",
        "https://www.expedia.com/explore/destinations",
        "https://www.kayak.com/explore",
        "https://www.trivago.com/",
        "https://www.hotels.com/",
        "https://www.airbnb.com/",
        "https://www.agoda.com/",
        "https://www.skyscanner.net/",
        "https://www.orbitz.com/",
        "https://www.priceline.com/",
        "https://www.travelocity.com/"
    ]

    # Crear instancia del crawler y configurar parámetros
    crawler = TourismCrawler(starting_urls)

    # Configuración específica para el crawler de turismo
    crawler.max_pages = 200  # Limitar a 200 páginas para evitar excesivo procesamiento
    crawler.max_depth = 2    # Profundidad de exploración moderada

    # Iniciar el proceso de crawling
    print("Iniciando extracción de información de turismo...")
    #pages_processed = crawler.run_crawler()
    #print(f"Proceso finalizado. Se procesaron {pages_processed} páginas.")

    # Crear instancia del sistema RAG
    rag_system = RAGSystem(crawler.collection)

    while True:
        user_query = input("\nIngrese una consulta para el sistema RAG o 'clean' para borrar el contexto (o 'salir' para terminar): ")
        if user_query.lower() == 'salir':
            break
        if user_query.lower() == 'clean':
            rag_system.answersContext = []
            continue

        response = rag_system.rag_query(user_query)
        print(f"Respuesta: {response}")
