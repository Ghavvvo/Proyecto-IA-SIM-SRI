"""
Versión 2: Agente de simulación intermedio
- Sin lógica difusa pero con cálculos más sofisticados
- Perfiles de turista básicos
- Considera cansancio pero de forma simple
- Considera preferencias del cliente de forma básica
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from autogen import Agent

class TouristSimulationAgentV2(Agent):
    def __init__(self, name: str, tourist_profile: str = "average"):
        """
        Versión intermedia del agente de simulación
        """
        super().__init__(name)
        self.tourist_profile = tourist_profile
        
        
        self.tourist_profiles = {
            "exigente": {"umbral_satisfaccion": 7.0, "sensibilidad": 0.8},
            "relajado": {"umbral_satisfaccion": 5.0, "sensibilidad": 0.5},
            "average": {"umbral_satisfaccion": 6.0, "sensibilidad": 0.65}
        }
        
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 5.5  
        self.lugares_visitados = []

    def _generar_clima(self, temporada: str = "verano", probabilidad_lluvia: float = 0.2) -> Tuple[str, float]:
        """Genera clima con más variación según temporada"""
        
        temp_bases = {
            "verano": 28,
            "invierno": 10,
            "otoño": 18,
            "primavera": 22
        }
        
        temp_base = temp_bases.get(temporada, 20)
        temperatura = temp_base + random.normalvariate(0, 3)
        
        
        llueve = random.random() < probabilidad_lluvia
        
        if llueve:
            valor_clima = random.uniform(2, 5)
            descripcion = f"Lluvia, {temperatura:.1f}°C"
        else:
            
            distancia_ideal = abs(temperatura - 24)
            valor_clima = max(3, 10 - distancia_ideal * 0.5)
            descripcion = f"Despejado, {temperatura:.1f}°C"
        
        return descripcion, round(valor_clima, 1)

    def _generar_crowding(self, lugar: Dict, dia_semana: str, hora: int) -> Tuple[str, float]:
        """Genera crowding considerando día y hora"""
        popularidad = lugar.get("popularidad", 5)
        
        
        modificador_dia = 1.0
        if dia_semana.lower() in ["sabado", "domingo"]:
            modificador_dia = 1.3
        elif dia_semana.lower() in ["lunes", "martes"]:
            modificador_dia = 0.7
        
        
        modificador_hora = 1.0
        if 11 <= hora <= 14 or 17 <= hora <= 19:
            modificador_hora = 1.4
        elif hora < 10 or hora > 20:
            modificador_hora = 0.5
        
        valor_crowding = popularidad * modificador_dia * modificador_hora
        valor_crowding += random.uniform(-1, 1)
        valor_crowding = max(0, min(10, valor_crowding))
        
        if valor_crowding < 4:
            descripcion = "Poco concurrido"
        elif valor_crowding < 7:
            descripcion = "Moderadamente lleno"
        else:
            descripcion = "Muy concurrido"
        
        return descripcion, round(valor_crowding, 1)

    def _generar_atencion(self, tipo_lugar: str) -> Tuple[str, float]:
        """Genera atención con variación por tipo de lugar"""
        niveles_base = {
            "museo": 7.0,
            "restaurante": 6.5,
            "parque": 5.5,
            "monumento": 6.0,
            "hotel": 7.5,
            "playa": 5.0,
            "centro_comercial": 6.0,
            "teatro": 7.0,
            "zoo": 6.0
        }
        
        nivel_base = niveles_base.get(tipo_lugar.lower(), 6.0)
        valor_atencion = nivel_base + random.normalvariate(0, 1)
        valor_atencion = max(0, min(10, valor_atencion))
        
        if valor_atencion < 5:
            descripcion = "Mala atención"
        elif valor_atencion < 7:
            descripcion = "Atención aceptable"
        else:
            descripcion = "Buena atención"
        
        return descripcion, round(valor_atencion, 1)

    def _generar_tiempo_espera(self, lugar: Dict, nivel_crowding: float) -> Tuple[str, float]:
        """Genera tiempo de espera basado en crowding y tipo de lugar"""
        tiempos_base = {
            "museo": 10,
            "restaurante": 15,
            "parque": 2,
            "monumento": 5,
            "hotel": 5,
            "playa": 1,
            "centro_comercial": 3,
            "teatro": 10,
            "zoo": 12
        }
        
        tiempo_base = tiempos_base.get(lugar.get("tipo", "otro").lower(), 8)
        
        
        factor_crowding = 1 + (nivel_crowding / 10) ** 1.5
        
        tiempo_espera = tiempo_base * factor_crowding * random.uniform(0.8, 1.2)
        tiempo_espera = max(0, min(90, tiempo_espera))
        
        if tiempo_espera < 10:
            descripcion = "Espera breve"
        elif tiempo_espera < 30:
            descripcion = "Espera moderada"
        else:
            descripcion = "Espera larga"
        
        return descripcion, round(tiempo_espera, 1)

    def _calcular_interes_lugar(self, lugar: Dict, preferencias_cliente: List[str] = None) -> float:
        """Calcula interés considerando preferencias básicas"""
        tipo_lugar = lugar.get("tipo", "otro").lower()
        
        
        tipo_a_categoria = {
            "museo": "cultura",
            "monumento": "cultura",
            "restaurante": "gastronomía",
            "parque": "naturaleza",
            "playa": "playa",
            "centro_comercial": "compras",
            "teatro": "cultura",
            "zoo": "naturaleza"
        }
        
        categoria_lugar = tipo_a_categoria.get(tipo_lugar, "otro")
        
        
        interes_base = 5.0
        
        if preferencias_cliente:
            
            for pref in preferencias_cliente:
                if categoria_lugar.lower() in pref.lower() or pref.lower() in categoria_lugar.lower():
                    interes_base = 7.5
                    break
        
        
        popularidad = lugar.get("popularidad", 5)
        interes = interes_base + (popularidad - 5) * 0.3
        
        
        interes += random.normalvariate(0, 1)
        
        return max(0, min(10, interes))

    def _actualizar_cansancio(self, tiempo_visita: float, distancia_km: float):
        """Actualiza cansancio de forma simple"""
        incremento = tiempo_visita * 0.5 + distancia_km * 0.2
        self.cansancio_acumulado += incremento
        self.cansancio_acumulado = min(10, self.cansancio_acumulado)

    def simular_visita(self, lugar: Dict, contexto: Dict) -> Dict:
        """Simula visita con cálculos intermedios"""
        
        clima_desc, valor_clima = self._generar_clima(
            contexto.get("temporada", "verano"),
            contexto.get("prob_lluvia", 0.2)
        )
        
        crowding_desc, valor_crowding = self._generar_crowding(
            lugar,
            contexto.get("dia_semana", "sabado"),
            contexto.get("hora", 14)
        )
        
        atencion_desc, valor_atencion = self._generar_atencion(lugar.get("tipo", "otro"))
        tiempo_espera_desc, minutos_espera = self._generar_tiempo_espera(lugar, valor_crowding)
        
        preferencias = contexto.get("preferencias_cliente", [])
        interes = self._calcular_interes_lugar(lugar, preferencias)
        
        
        perfil = self.tourist_profiles[self.tourist_profile]
        sensibilidad = perfil["sensibilidad"]
        
        
        satisfaccion_lugar = (
            interes * 0.35 +  
            valor_clima * 0.20 * sensibilidad +
            valor_atencion * 0.20 +
            (10 - valor_crowding) * 0.15 * sensibilidad +
            (10 - minutos_espera/9) * 0.10 * sensibilidad
        )
        
        
        factor_cansancio = max(0.8, 1 - self.cansancio_acumulado / 20)
        satisfaccion_lugar *= factor_cansancio
        
        
        satisfaccion_lugar *= 0.92  
        satisfaccion_lugar = max(0, min(10, satisfaccion_lugar))
        
        
        tiempos_base = {
            "museo": 1.5,
            "restaurante": 1.2,
            "parque": 1.0,
            "monumento": 0.5,
            "playa": 2.0,
            "centro_comercial": 1.5,
            "teatro": 2.0,
            "zoo": 2.5
        }
        tiempo_base = tiempos_base.get(lugar.get("tipo", "otro").lower(), 1.0)
        factor_interes = 0.7 + (interes / 10) * 0.6
        tiempo_visita = tiempo_base * factor_interes
        
        
        self._actualizar_cansancio(tiempo_visita, contexto.get("distancia_km", 2))
        
        
        if self.lugares_visitados:
            peso = 1 / (len(self.lugares_visitados) + 1)
            self.satisfaccion_general = self.satisfaccion_general * (1 - peso) + satisfaccion_lugar * peso
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
            "tiempo_visita_hrs": round(tiempo_visita, 2),
            "interes": round(interes, 1),
            "satisfaccion": round(satisfaccion_lugar, 1),
            "cansancio": round(self.cansancio_acumulado, 1)
        }
        
        self.lugares_visitados.append(visita)
        
        
        umbral = perfil["umbral_satisfaccion"]
        if satisfaccion_lugar > umbral + 1:
            visita["comentario"] = f"Excelente experiencia en {lugar.get('nombre')}. Superó expectativas."
        elif satisfaccion_lugar > umbral - 1:
            visita["comentario"] = f"Buena visita a {lugar.get('nombre')}. Cumplió con lo esperado."
        else:
            visita["comentario"] = f"Experiencia decepcionante en {lugar.get('nombre')}. Por debajo de expectativas."
        
        return visita

    def simular_itinerario(self, itinerario: List[Dict], contexto_base: Dict) -> Dict:
        """Simula itinerario considerando días pero sin descanso nocturno"""
        
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 5.5  
        self.lugares_visitados = []
        
        dia_actual = 1
        hora_actual = contexto_base.get("hora_inicio", 9)
        lugares_por_dia = {}
        
        for i, lugar in enumerate(itinerario):
            dia_lugar = lugar.get("dia", dia_actual)
            
            if dia_lugar > dia_actual:
                
                self.cansancio_acumulado *= 0.7
                dia_actual = dia_lugar
                hora_actual = contexto_base.get("hora_inicio", 9)
            
            contexto_lugar = contexto_base.copy()
            contexto_lugar["hora"] = int(hora_actual)
            contexto_lugar["dia_actual"] = dia_actual
            
            if i > 0 and lugar.get("dia", dia_actual) == itinerario[i-1].get("dia", dia_actual):
                contexto_lugar["distancia_km"] = lugar.get("distancia_anterior", 2)
            else:
                contexto_lugar["distancia_km"] = lugar.get("distancia_inicio", 5)
            
            resultado_visita = self.simular_visita(lugar, contexto_lugar)
            resultado_visita["dia"] = dia_actual
            
            if dia_actual not in lugares_por_dia:
                lugares_por_dia[dia_actual] = []
            lugares_por_dia[dia_actual].append(resultado_visita["lugar"])
            
            
            tiempo_total = resultado_visita["tiempo_espera_min"] / 60 + resultado_visita["tiempo_visita_hrs"]
            hora_actual += tiempo_total
        
        
        duracion_total_hrs = sum(v["tiempo_visita_hrs"] + v["tiempo_espera_min"]/60 for v in self.lugares_visitados)
        
        
        resultados = {
            "perfil_turista": self.tourist_profile,
            "lugares_visitados": self.lugares_visitados,
            "satisfaccion_general": round(self.satisfaccion_general, 1),
            "cansancio_final": round(self.cansancio_acumulado, 1),
            "duracion_total_hrs": duracion_total_hrs,
            "dias_simulados": dia_actual,
            "lugares_por_dia": lugares_por_dia,
            "valoracion_viaje": self._generar_valoracion_final()
        }
        
        return resultados

    def _generar_valoracion_final(self) -> str:
        """Genera valoración según perfil"""
        perfil = self.tourist_profiles[self.tourist_profile]
        umbral = perfil["umbral_satisfaccion"]
        
        if self.satisfaccion_general >= umbral + 1.5:
            return f"Experiencia excepcional con satisfacción de {self.satisfaccion_general}/10. El viaje superó ampliamente las expectativas del perfil {self.tourist_profile}."
        elif self.satisfaccion_general >= umbral:
            return f"Viaje satisfactorio con puntuación de {self.satisfaccion_general}/10. Cumplió con las expectativas del perfil {self.tourist_profile}."
        else:
            return f"Experiencia por debajo de expectativas con {self.satisfaccion_general}/10. No alcanzó el umbral de satisfacción para el perfil {self.tourist_profile}."

    def receive(self, message: Dict, sender: Agent) -> Dict:
        """Procesa mensajes"""
        msg_type = message.get('type')
        
        if msg_type == 'simulate_itinerary':
            profile = message.get('profile')
            if profile and profile in self.tourist_profiles:
                self.tourist_profile = profile
            
            return {
                'type': 'simulation_results',
                'results': self.simular_itinerario(
                    message.get('itinerary', []),
                    message.get('context', {})
                )
            }
        
        return {'type': 'error', 'msg': 'Mensaje no soportado'}