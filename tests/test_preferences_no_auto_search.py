"""
Test para verificar que el sistema no busca automáticamente en DuckDuckGo
al obtener las preferencias del usuario
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_coordinator import CoordinatorAgent
from agent_crawler import CrawlerAgent
from agent_rag import RAGAgent
from agent_interface import InterfaceAgent
from agent_context import ContextAgent
from agent_route import RouteAgent
from agent_tourist_guide import TouristGuideAgent
from crawler import TourismCrawler
from rag import EnhancedRAGSystem
from urls import starting_urls
import time

def test_preferences_flow():
    """
    Prueba el flujo de preferencias para verificar que:
    1. Primero intenta usar la información de la BD local
    2. Solo busca en DuckDuckGo si la información no es suficiente
    """
    print("🧪 Iniciando prueba de flujo de preferencias sin búsqueda automática...")
    print("=" * 80)
    
    
    crawler_agent = CrawlerAgent(
        name="crawler_agent", 
        starting_urls=starting_urls, 
        max_pages=200, 
        max_depth=2,
        num_threads=10
    )
    rag_agent = RAGAgent("rag_agent")
    interface_agent = InterfaceAgent("interface_agent")
    context_agent = ContextAgent("context_agent")
    route_agent = RouteAgent("route_agent")
    tourist_guide_agent = TouristGuideAgent("tourist_guide_agent")
    
    coordinator = CoordinatorAgent(
        "coordinator",
        crawler_agent,
        rag_agent,
        interface_agent,
        context_agent,
        route_agent,
        tourist_guide_agent
    )
    
    
    print("🚀 Inicializando sistema...")
    coordinator.start()
    time.sleep(2)
    
    
    print("\n📝 Iniciando planificación de vacaciones...")
    response = coordinator.ask("quiero planificar vacaciones")
    print(f"\n🤖 Sistema: {response}")
    
    
    print("\n👤 Usuario: Cuba")
    response = coordinator.ask("Cuba")
    print(f"\n🤖 Sistema: {response}")
    
    print("\n👤 Usuario: playas y museos")
    response = coordinator.ask("playas y museos")
    print(f"\n🤖 Sistema: {response}")
    
    
    print("\n👤 Usuario: con eso es suficiente, genera el itinerario")
    print("\n⏳ Observando el comportamiento del sistema...")
    print("   - Debería primero consultar la BD local")
    print("   - Solo si no hay información suficiente, buscar en DuckDuckGo")
    print("\n" + "=" * 80)
    
    response = coordinator.ask("con eso es suficiente, genera el itinerario")
    print(f"\n🤖 Sistema: {response[:500]}...")  
    
    print("\n" + "=" * 80)
    print("✅ Prueba completada")
    print("\nVerifica en los logs de arriba:")
    print("1. Si aparece '📚 Consultando información existente en la base de datos...'")
    print("2. Si aparece '✅ Encontré suficiente información en la base de datos local' (no busca en DuckDuckGo)")
    print("3. O si aparece '⚠️ La información en la base de datos no es suficiente' (entonces sí busca)")

if __name__ == "__main__":
    test_preferences_flow()