from agent_crawler import CrawlerAgent
from agent_rag import RAGAgent
from agent_coordinator import CoordinatorAgent
from agent_interface import InterfaceAgent
from urls import starting_urls

if __name__ == "__main__":
    # Crear agentes con crawler paralelo
    print("🚀 Configurando sistema con crawler paralelo...")
    
    crawler_agent = CrawlerAgent(
        name="crawler_agent", 
        starting_urls=starting_urls, 
        max_pages=200, 
        max_depth=2,
        num_threads=10  # 10 hilos en paralelo
    )
    
    rag_agent = RAGAgent("rag_agent")
    interface_agent = InterfaceAgent("interface_agent")
    coordinator = CoordinatorAgent("coordinator", crawler_agent, rag_agent, interface_agent)

    # Iniciar el sistema multiagente
    print("⚡ Iniciando sistema multiagente de turismo con crawler paralelo...")
    print(f"🔧 Configuración: {crawler_agent.crawler.num_threads} hilos paralelos")
    coordinator.start()

    while True:
        user_query = input("\n🔍 Ingrese una consulta para el sistema RAG o 'salir' para terminar: ")
        if user_query.lower() == 'salir':
            print("👋 ¡Hasta luego!")
            break
        
        print(f"🤔 Procesando consulta: {user_query}")
        response = coordinator.ask(user_query)
        print(f"✅ Respuesta: {response}")
