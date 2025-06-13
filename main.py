from agent_crawler import CrawlerAgent
from agent_rag import RAGAgent
from agent_coordinator import CoordinatorAgent
from agent_interface import InterfaceAgent
from urls import starting_urls

if __name__ == "__main__":
    # Crear agentes
    crawler_agent = CrawlerAgent("crawler_agent", starting_urls, max_pages=200, max_depth=2)
    rag_agent = RAGAgent("rag_agent")
    interface_agent = InterfaceAgent("interface_agent")
    coordinator = CoordinatorAgent("coordinator", crawler_agent, rag_agent, interface_agent)

    # Iniciar el sistema multiagente
    print("Iniciando sistema multiagente de turismo...")
    coordinator.start()

    while True:
        user_query = input("\nIngrese una consulta para el sistema RAG o 'salir' para terminar: ")
        if user_query.lower() == 'salir':
            break
        response = coordinator.ask(user_query)
        print(f"Respuesta: {response}")
