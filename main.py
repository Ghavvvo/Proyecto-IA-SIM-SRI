from agent_crawler import CrawlerAgent
from agent_rag import RAGAgent
from agent_coordinator import CoordinatorAgent
from agent_interface import InterfaceAgent
from agent_context import ContextAgent
from urls import starting_urls

if __name__ == "__main__":
    # Crear agentes con crawler paralelo y contexto conversacional
    print("🚀 Configurando sistema con crawler paralelo y contexto conversacional...")
    
    crawler_agent = CrawlerAgent(
        name="crawler_agent", 
        starting_urls=starting_urls, 
        max_pages=200, 
        max_depth=2,
        num_threads=10  # 10 hilos en paralelo
    )
    
    rag_agent = RAGAgent("rag_agent")
    interface_agent = InterfaceAgent("interface_agent")
    context_agent = ContextAgent("context_agent")
    coordinator = CoordinatorAgent("coordinator", crawler_agent, rag_agent, interface_agent, context_agent)

    # Iniciar el sistema multiagente
    print("⚡ Iniciando sistema multiagente de turismo con crawler paralelo...")
    print(f"🔧 Configuración: {crawler_agent.crawler.num_threads} hilos paralelos")
    coordinator.start()

    print("\n📋 Comandos disponibles:")
    print("  - Escriba su consulta normalmente")
    print("  - 'stats' - Ver estadísticas de conversación")
    print("  - 'contexto' - Ver historial de conversación")
    print("  - 'limpiar' - Limpiar contexto de conversación")
    print("  - 'salir' - Terminar el programa")

    while True:
        user_query = input("\n🔍 Ingrese una consulta para el sistema RAG: ")
        
        if user_query.lower() == 'salir':
            print("👋 ¡Hasta luego!")
            break
        elif user_query.lower() == 'stats':
            stats = coordinator.get_conversation_stats()
            print("\n📊 Estadísticas de conversación:")
            print(f"  - Total de interacciones: {stats['total_interactions']}")
            print(f"  - Longitud promedio de consultas: {stats['average_query_length']} caracteres")
            print(f"  - Longitud promedio de respuestas: {stats['average_response_length']} caracteres")
            print(f"  - Duración de conversación: {stats['conversation_duration']}")
            if stats['most_recent_topic']:
                print(f"  - Tema más reciente: {stats['most_recent_topic'][:100]}...")
            continue
        elif user_query.lower() == 'contexto':
            context = coordinator.get_conversation_context()
            if context and context['interaction_count'] > 0:
                print(f"\n💬 Historial de conversación ({context['interaction_count']} interacciones):")
                for i, interaction in enumerate(context['history'][-3:], 1):  # Mostrar últimas 3
                    print(f"\n  {i}. Usuario: {interaction['query']}")
                    print(f"     Sistema: {interaction['response'][:150]}...")
            else:
                print("\n💬 No hay historial de conversación disponible")
            continue
        elif user_query.lower() == 'limpiar':
            if coordinator.clear_conversation_context():
                print("✅ Contexto de conversación limpiado")
            else:
                print("❌ Error al limpiar el contexto")
            continue
        
        print(f"🤔 Procesando consulta: {user_query}")
        response = coordinator.ask(user_query)
        print(f"✅ Respuesta: {response}")
