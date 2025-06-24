from agents.agent_crawler import CrawlerAgent
from agents.agent_rag import RAGAgent
from agents.agent_coordinator import CoordinatorAgent
from agents.agent_interface import InterfaceAgent
from agents.agent_context import ContextAgent
from agents.agent_route import RouteAgent
from agents.agent_tourist_guide import TouristGuideAgent
from agents.agent_simulation import TouristSimulationAgent
from utils.urls import starting_urls
from dotenv import load_dotenv

# Suprimir warning de flaml.automl
import warnings

# Suprimir warning de flaml.automl
import warnings
warnings.filterwarnings('ignore', message='flaml.automl is not available')

warnings.filterwarnings('ignore', message='flaml.automl is not available')

if __name__ == "__main__":
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar que se cargó la API key
    import os
    if os.getenv('MISTRAL_API_KEY'):
        print("✅ MISTRAL_API_KEY cargada correctamente")
    else:
        print("❌ Error: MISTRAL_API_KEY no encontrada en las variables de entorno")
        print("   Asegúrese de que el archivo .env existe y contiene MISTRAL_API_KEY=su_clave_aqui")
    
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
    route_agent = RouteAgent("route_agent")
    tourist_guide_agent = TouristGuideAgent("tourist_guide_agent")
    simulation_agent = TouristSimulationAgent("simulation_agent", "average")  # Perfil por defecto: average
    coordinator = CoordinatorAgent("coordinator", crawler_agent, rag_agent, interface_agent, context_agent, route_agent, tourist_guide_agent, simulation_agent)

    # Iniciar el sistema multiagente
    print("⚡ Iniciando sistema multiagente de turismo con crawler paralelo...")
    print(f"🔧 Configuración: {crawler_agent.crawler.num_threads} hilos paralelos")
    print("🎮 Agente de simulación turística activado con lógica difusa")
    coordinator.start()

    print("\n📋 Comandos disponibles durante la conversación:")
    print("  - 'stats' - Ver estadísticas de conversación")
    print("  - 'contexto' - Ver historial de conversación")
    print("  - 'limpiar' - Limpiar contexto de conversación")
    print("  - 'salir' - Terminar el programa")

    
    # Iniciar directamente con el asistente de planificación de vacaciones
    print("\n" + "="*60)
    print("🏖️ ¡Bienvenido al Asistente de Planificación de Vacaciones!")
    print("="*60 + "\n")
    
    # Activar modo planificación automáticamente
    initial_response = coordinator._start_vacation_planning()
    print(f"🤖 {initial_response}")

    while True:
        user_query = input("\n👤 Tu respuesta: ")
        
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
        
        # No mostrar "Procesando consulta" para comandos del sistema
        if user_query.lower() not in ['stats', 'contexto', 'limpiar', 'salir']:
            response = coordinator.ask(user_query)
            print(f"\n🤖 {response}")
