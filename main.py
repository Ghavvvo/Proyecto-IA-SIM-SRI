from crawler import TourismCrawler
from rag import RAGSystem
from urls import starting_urls


if __name__ == "__main__":

    crawler = TourismCrawler(starting_urls, max_pages=200, max_depth=2)
    print("Iniciando extracción de información de turismo...")
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
