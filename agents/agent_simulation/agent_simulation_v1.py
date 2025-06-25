"""
Versión 1: Agente de simulación básico
- Sin lógica difusa
- Cálculos simples basados en promedios
- Sin perfiles de turista diferenciados
- Sin consideración de cansancio acumulado
"""

import random
from typing import Dict, List, Tuple
from autogen import Agent

class TouristSimulationAgentV1(Agent):
    def __init__(self, name: str, tourist_profile: str = "average"):
        """
        Versión simplificada del agente de simulación
        """
        super().__init__(name)
        self.tourist_profile = tourist_profile
        self.satisfaccion_general = 4.5  
        self.lugares_visitados = []

    def _generar_clima(self, temporada: str = "verano") -> Tuple[str, float]:
        """Genera clima aleatorio simple"""
        if temporada == "verano":
            valor_clima = random.uniform(6, 9)
        elif temporada == "invierno":
            valor_clima = random.uniform(3, 7)
        else:
            valor_clima = random.uniform(5, 8)
        
        descripcion = "Buen clima" if valor_clima > 6 else "Clima regular"
        return descripcion, round(valor_clima, 1)

    def _generar_crowding(self, lugar: Dict) -> Tuple[str, float]:
        """Genera nivel de aglomeración simple"""
        popularidad = lugar.get("popularidad", 5)
        valor_crowding = popularidad + random.uniform(-2, 2)
        valor_crowding = max(0, min(10, valor_crowding))
        
        descripcion = "Muy lleno" if valor_crowding > 7 else "Normal"
        return descripcion, round(valor_crowding, 1)

    def _generar_atencion(self, tipo_lugar: str) -> Tuple[str, float]:
        """Genera calidad de atención simple"""
        
        valor_atencion = random.uniform(5, 8)
        descripcion = "Buena atención" if valor_atencion > 6.5 else "Atención regular"
        return descripcion, round(valor_atencion, 1)

    def _generar_tiempo_espera(self, valor_crowding: float) -> Tuple[str, float]:
        """Genera tiempo de espera simple"""
        
        tiempo_espera = valor_crowding * 5 + random.uniform(-10, 10)
        tiempo_espera = max(0, min(60, tiempo_espera))
        
        descripcion = "Espera larga" if tiempo_espera > 30 else "Espera corta"
        return descripcion, round(tiempo_espera, 1)

    def _calcular_interes_lugar(self, lugar: Dict) -> float:
        """Calcula interés simple basado solo en popularidad"""
        popularidad = lugar.get("popularidad", 5)
        interes = popularidad + random.uniform(-1, 1)
        return max(0, min(10, interes))

    def simular_visita(self, lugar: Dict, contexto: Dict) -> Dict:
        """Simula visita con cálculos simples"""
        
        clima_desc, valor_clima = self._generar_clima(contexto.get("temporada", "verano"))
        crowding_desc, valor_crowding = self._generar_crowding(lugar)
        atencion_desc, valor_atencion = self._generar_atencion(lugar.get("tipo", "otro"))
        tiempo_espera_desc, minutos_espera = self._generar_tiempo_espera(valor_crowding)
        interes = self._calcular_interes_lugar(lugar)
        
        
        satisfaccion_lugar = (
            valor_clima * 0.2 +
            (10 - valor_crowding) * 0.2 +  
            valor_atencion * 0.2 +
            (10 - minutos_espera/6) * 0.2 +  
            interes * 0.2
        )
        
        satisfaccion_lugar *= 0.85  
        satisfaccion_lugar = max(0, min(10, satisfaccion_lugar))
        
        
        tiempos_visita = {
            "museo": 1.5,
            "restaurante": 1.0,
            "parque": 1.0,
            "monumento": 0.5,
            "playa": 2.0,
            "centro_comercial": 1.5,
            "teatro": 2.0,
            "zoo": 2.5
        }
        tiempo_visita = tiempos_visita.get(lugar.get("tipo", "otro").lower(), 1.0)
        
        
        if self.lugares_visitados:
            self.satisfaccion_general = (self.satisfaccion_general + satisfaccion_lugar) / 2
        else:
            self.satisfaccion_general = satisfaccion_lugar
        
        
        visita = {
            "lugar": lugar.get("nombre", "Lugar sin nombre"),
            "tipo": lugar.get("tipo", "otro"),
            "clima": clima_desc,
            "crowding": crowding_desc,
            "atencion": atencion_desc,
            "tiempo_espera": tiempo_espera_desc,
            "tiempo_espera_min": minutos_espera,
            "tiempo_visita_hrs": tiempo_visita,
            "interes": round(interes, 1),
            "satisfaccion": round(satisfaccion_lugar, 1),
            "cansancio": 0  
        }
        
        self.lugares_visitados.append(visita)
        
        
        if satisfaccion_lugar > 7:
            visita["comentario"] = f"Buena experiencia en {lugar.get('nombre')}."
        elif satisfaccion_lugar > 5:
            visita["comentario"] = f"Experiencia regular en {lugar.get('nombre')}."
        else:
            visita["comentario"] = f"Mala experiencia en {lugar.get('nombre')}."
        
        return visita

    def simular_itinerario(self, itinerario: List[Dict], contexto_base: Dict) -> Dict:
        """Simula itinerario completo sin considerar días ni cansancio"""
        
        self.satisfaccion_general = 4.5
        self.lugares_visitados = []
        
        for lugar in itinerario:
            self.simular_visita(lugar, contexto_base)
        
        
        duracion_total_hrs = sum(v["tiempo_visita_hrs"] for v in self.lugares_visitados)
        
        
        resultados = {
            "perfil_turista": self.tourist_profile,
            "lugares_visitados": self.lugares_visitados,
            "satisfaccion_general": round(self.satisfaccion_general, 1),
            "cansancio_final": 0,  
            "duracion_total_hrs": duracion_total_hrs,
            "dias_simulados": 1,  
            "lugares_por_dia": {1: [v["lugar"] for v in self.lugares_visitados]},
            "valoracion_viaje": self._generar_valoracion_final()
        }
        
        return resultados

    def _generar_valoracion_final(self) -> str:
        """Genera valoración simple"""
        if self.satisfaccion_general >= 7:
            return f"Buen viaje con satisfacción de {self.satisfaccion_general}/10."
        elif self.satisfaccion_general >= 5:
            return f"Viaje regular con satisfacción de {self.satisfaccion_general}/10."
        else:
            return f"Mal viaje con satisfacción de {self.satisfaccion_general}/10."

    def receive(self, message: Dict, sender: Agent) -> Dict:
        """Procesa mensajes básicos"""
        msg_type = message.get('type')
        
        if msg_type == 'simulate_itinerary':
            return {
                'type': 'simulation_results',
                'results': self.simular_itinerario(
                    message.get('itinerary', []),
                    message.get('context', {})
                )
            }
        
        return {'type': 'error', 'msg': 'Mensaje no soportado'}