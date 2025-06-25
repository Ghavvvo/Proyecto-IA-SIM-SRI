import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Tuple, Any
from autogen import Agent


import os
import sys

if any('experiment' in arg for arg in sys.argv) or os.environ.get('TESTING', '').lower() == 'true':
    import matplotlib
    matplotlib.use('Agg')  

import matplotlib.pyplot as plt

class TouristSimulationAgent(Agent):
    def __init__(self, name: str, tourist_profile: str = "average"):
        """
        Agente de simulación de turistas que evalúa experiencias usando variables aleatorias y lógica difusa

        Args:
            name: Identificador único del agente
            tourist_profile: Perfil del turista ("exigente", "relajado", "average")
        """
        super().__init__(name)
        self.tourist_profile = tourist_profile

        
        self.tourist_profiles = {
            "exigente": {
                "umbral_satisfaccion": 7.5,
                "sensibilidad_clima": 0.8,
                "sensibilidad_crowding": 0.9,
                "sensibilidad_atencion": 0.9,
                "sensibilidad_tiempo": 0.7,
                "paciencia": 3
            },
            "relajado": {
                "umbral_satisfaccion": 5.0,
                "sensibilidad_clima": 0.4,
                "sensibilidad_crowding": 0.3,
                "sensibilidad_atencion": 0.6,
                "sensibilidad_tiempo": 0.3,
                "paciencia": 8
            },
            "average": {
                "umbral_satisfaccion": 6.0,
                "sensibilidad_clima": 0.6,
                "sensibilidad_crowding": 0.6,
                "sensibilidad_atencion": 0.7,
                "sensibilidad_tiempo": 0.5,
                "paciencia": 5
            }
        }

        
        self._setup_fuzzy_system()

        
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 7.5  
        self.lugares_visitados = []

    def _setup_fuzzy_system(self):
        """Configurar el sistema de lógica difusa para evaluación de experiencias"""
        
        self.clima = ctrl.Antecedent(np.arange(0, 11, 1), 'clima')
        self.crowding = ctrl.Antecedent(np.arange(0, 11, 1), 'crowding')
        self.atencion = ctrl.Antecedent(np.arange(0, 11, 1), 'atencion')
        self.tiempo_espera = ctrl.Antecedent(np.arange(0, 121, 1), 'tiempo_espera')
        self.interes_lugar = ctrl.Antecedent(np.arange(0, 11, 1), 'interes_lugar')

        
        self.satisfaccion = ctrl.Consequent(np.arange(0, 11, 1), 'satisfaccion')

        
        
        self.clima['malo'] = fuzz.trimf(self.clima.universe, [0, 0, 5])
        self.clima['moderado'] = fuzz.trimf(self.clima.universe, [2, 5, 8])
        self.clima['bueno'] = fuzz.trimf(self.clima.universe, [5, 10, 10])

        
        self.crowding['bajo'] = fuzz.trimf(self.crowding.universe, [0, 0, 4])
        self.crowding['medio'] = fuzz.trimf(self.crowding.universe, [3, 5, 7])
        self.crowding['alto'] = fuzz.trimf(self.crowding.universe, [6, 10, 10])

        
        self.atencion['mala'] = fuzz.trimf(self.atencion.universe, [0, 0, 4])
        self.atencion['regular'] = fuzz.trimf(self.atencion.universe, [3, 5, 7])
        self.atencion['buena'] = fuzz.trimf(self.atencion.universe, [6, 10, 10])

        
        self.tiempo_espera['corto'] = fuzz.trimf(self.tiempo_espera.universe, [0, 0, 20])
        self.tiempo_espera['medio'] = fuzz.trimf(self.tiempo_espera.universe, [15, 30, 60])
        self.tiempo_espera['largo'] = fuzz.trimf(self.tiempo_espera.universe, [45, 120, 120])

        
        self.interes_lugar['bajo'] = fuzz.trimf(self.interes_lugar.universe, [0, 0, 4])
        self.interes_lugar['medio'] = fuzz.trimf(self.interes_lugar.universe, [3, 5, 7])
        self.interes_lugar['alto'] = fuzz.trimf(self.interes_lugar.universe, [6, 10, 10])

        
        self.satisfaccion['baja'] = fuzz.trimf(self.satisfaccion.universe, [0, 0, 4])
        self.satisfaccion['media'] = fuzz.trimf(self.satisfaccion.universe, [3, 5, 7])
        self.satisfaccion['alta'] = fuzz.trimf(self.satisfaccion.universe, [6, 10, 10])

        
        regla1 = ctrl.Rule(
            self.clima['bueno'] & self.crowding['bajo'] & self.atencion['buena'] &
            self.tiempo_espera['corto'] & self.interes_lugar['alto'],
            self.satisfaccion['alta']
        )

        regla2 = ctrl.Rule(
            self.clima['malo'] & self.crowding['alto'] & self.tiempo_espera['largo'],
            self.satisfaccion['baja']
        )

        regla3 = ctrl.Rule(
            self.interes_lugar['alto'] & self.atencion['buena'],
            self.satisfaccion['alta']
        )

        regla4 = ctrl.Rule(
            self.clima['moderado'] & self.crowding['medio'] & self.tiempo_espera['medio'],
            self.satisfaccion['media']
        )

        regla5 = ctrl.Rule(
            self.interes_lugar['bajo'],
            self.satisfaccion['baja']
        )

        regla6 = ctrl.Rule(
            self.atencion['mala'] & self.tiempo_espera['largo'],
            self.satisfaccion['baja']
        )

        regla7 = ctrl.Rule(
            self.clima['bueno'] & self.interes_lugar['alto'] & self.tiempo_espera['corto'],
            self.satisfaccion['alta']
        )

        regla8 = ctrl.Rule(
            self.crowding['alto'] & self.tiempo_espera['largo'] & self.interes_lugar['medio'],
            self.satisfaccion['media']
        )

        
        self.system = ctrl.ControlSystem([
            regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8
        ])

        
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def _generar_clima(self, temporada: str = "verano", probabilidad_lluvia: float = 0.2) -> Tuple[str, float]:
        """
        Genera condiciones climáticas aleatorias basadas en la temporada

        Args:
            temporada: Temporada del año (verano, invierno, otoño, primavera)
            probabilidad_lluvia: Probabilidad base de lluvia

        Returns:
            Tupla con (descripción del clima, valor numérico para el sistema difuso)
        """
        
        if temporada == "verano":
            temp_base = 28
            temp_var = 5
            prob_lluvia = probabilidad_lluvia * 0.4  
        elif temporada == "invierno":
            temp_base = 10
            temp_var = 8
            prob_lluvia = probabilidad_lluvia * 1.3  
        elif temporada == "otoño":
            temp_base = 18
            temp_var = 7
            prob_lluvia = probabilidad_lluvia * 1.1  
        else:  
            temp_base = 22
            temp_var = 6
            prob_lluvia = probabilidad_lluvia * 0.9  

        
        temperatura = round(random.normalvariate(temp_base, temp_var/3), 1)

        
        esta_lloviendo = random.random() < prob_lluvia

        
        if esta_lloviendo:
            valor_clima = max(2, 10 - random.randint(4, 8))  
            descripcion = f"Lluvia, {temperatura}°C"
        else:
            
            if temporada == "verano":
                if temperatura < 18:
                    valor_clima = random.uniform(4.5, 7.5)  
                elif temperatura > 35:
                    valor_clima = random.uniform(3.5, 6.5)  
                else:
                    valor_clima = min(10, max(5.5, 10 - abs(temperatura - 26) * random.uniform(0.3, 0.6)))  
            elif temporada == "invierno":
                if temperatura < 0:
                    valor_clima = random.uniform(3.0, 6.5)  
                else:
                    valor_clima = min(10, max(5.5, 10 - abs(temperatura - 12) * random.uniform(0.3, 0.6)))  
            else:  
                valor_clima = min(10, max(5.5, 10 - abs(temperatura - 22) * random.uniform(0.3, 0.6)))  

            descripcion = f"Despejado, {temperatura}°C"

        return descripcion, round(valor_clima, 1)

    def _generar_crowding(self, lugar: Dict, dia_semana: str, hora: int) -> Tuple[str, float]:
        """
        Genera nivel de aglomeración basado en popularidad del lugar, día y hora

        Args:
            lugar: Diccionario con información del lugar
            dia_semana: Día de la semana
            hora: Hora del día (0-23)

        Returns:
            Tupla con (descripción del crowding, valor numérico para el sistema difuso)
        """
        
        popularidad_base = lugar.get("popularidad", 5)

        
        modificador_dia = {
            "lunes": 0.65,   
            "martes": 0.65,  
            "miercoles": 0.75, 
            "jueves": 0.85,  
            "viernes": 1.0,
            "sabado": 1.4,   
            "domingo": 1.25  
        }.get(dia_semana.lower(), 1.0)

        
        if hora < 9 or hora > 20:
            modificador_hora = 0.45  
        elif 11 <= hora <= 14 or 17 <= hora <= 19:  
            modificador_hora = 1.4  
        else:
            modificador_hora = 1.0

        
        nivel_base = popularidad_base * modificador_dia * modificador_hora
        variacion = random.uniform(-2.5, 2.0)  
        nivel_crowding = max(0, min(10, nivel_base + variacion))

        
        if nivel_crowding < 3:
            descripcion = "Casi vacío"
        elif nivel_crowding < 5:
            descripcion = "Poco concurrido"
        elif nivel_crowding < 7:
            descripcion = "Moderadamente lleno"
        elif nivel_crowding < 9:
            descripcion = "Muy concurrido"
        else:
            descripcion = "Extremadamente lleno"

        return descripcion, round(nivel_crowding, 1)

    def _generar_atencion(self, tipo_lugar: str) -> Tuple[str, float]:
        """
        Genera calidad de atención al cliente según tipo de lugar

        Args:
            tipo_lugar: Categoría del lugar turístico

        Returns:
            Tupla con (descripción de la atención, valor numérico para el sistema difuso)
        """
        
        niveles_base = {
            "museo": 7.8,      
            "restaurante": 7.4, 
            "parque": 6.8,     
            "monumento": 7.4,  
            "hotel": 7.8,     
            "playa": 6.4,     
            "centro_comercial": 6.8, 
            "teatro": 7.4,    
            "zoo": 7.4,       
            "desplazamiento": 6.2,  
            "otro": 6.8       
        }

        nivel_base = niveles_base.get(tipo_lugar.lower(), 6.8)  

        
        variacion = random.normalvariate(0, 1.8)  
        nivel_atencion = max(2.5, min(10, nivel_base + variacion))

        
        if nivel_atencion < 3:
            descripcion = "Pésima atención"
        elif nivel_atencion < 5:
            descripcion = "Mala atención"
        elif nivel_atencion < 7:
            descripcion = "Atención aceptable"
        elif nivel_atencion < 9:
            descripcion = "Buena atención"
        else:
            descripcion = "Excelente atención"

        return descripcion, round(nivel_atencion, 1)

    def _generar_tiempo_espera(self, lugar: Dict, nivel_crowding: float) -> Tuple[str, float]:
        """
        Genera tiempo de espera basado en el tipo de lugar y nivel de aglomeración

        Args:
            lugar: Diccionario con información del lugar
            nivel_crowding: Valor de crowding (0-10)

        Returns:
            Tupla con (descripción del tiempo de espera, valor en minutos)
        """
        tipo_lugar = lugar.get("tipo", "otro")

        
        tiempos_base = {
            "museo": 12,       
            "restaurante": 18, 
            "parque": 3,       
            "monumento": 7,    
            "hotel": 6,        
            "playa": 2,        
            "centro_comercial": 5, 
            "teatro": 10,      
            "zoo": 12,         
            "desplazamiento": 3, 
            "otro": 7          
        }

        tiempo_base = tiempos_base.get(tipo_lugar.lower(), 7)

        
        factor_crowding = (nivel_crowding / 5.0) ** 1.6

        
        tiempo_espera = tiempo_base * factor_crowding * random.uniform(0.7, 1.2)  
        tiempo_espera = max(0, min(100, tiempo_espera))  

        
        if tiempo_espera < 5:
            descripcion = "Sin espera"
        elif tiempo_espera < 15:
            descripcion = "Espera breve"
        elif tiempo_espera < 30:
            descripcion = "Espera moderada"
        elif tiempo_espera < 60:
            descripcion = "Espera larga"
        else:
            descripcion = "Espera excesiva"

        return descripcion, round(tiempo_espera, 1)

    def _generar_tiempo_visita(self, lugar: Dict, nivel_interes: float) -> float:
        """
        Genera tiempo de visita basado en tipo de lugar e interés del turista

        Args:
            lugar: Diccionario con información del lugar
            nivel_interes: Nivel de interés del turista (0-10)

        Returns:
            Tiempo de visita en horas
        """
        tipo_lugar = lugar.get("tipo", "otro")

        
        tiempos_base = {
            "museo": 1.5,
            "restaurante": 1.2,
            "parque": 1.0,
            "monumento": 0.5,
            "hotel": 0.2,  
            "playa": 2.0,
            "centro_comercial": 1.5,
            "teatro": 2.0,
            "zoo": 2.5,
            "desplazamiento": 0.5,
            "otro": 1.0
        }

        tiempo_base = tiempos_base.get(tipo_lugar.lower(), 1.0)

        
        factor_interes = 0.65 + (nivel_interes / 10) * 0.7  

        
        variacion = random.normalvariate(0, 0.18)  

        tiempo_visita = tiempo_base * factor_interes + variacion
        return max(0.1, tiempo_visita)  

    def _actualizar_cansancio(self, tiempo_visita: float, distancia_km: float):
        """
        Actualiza el nivel de cansancio acumulado

        Args:
            tiempo_visita: Tiempo de visita en horas
            distancia_km: Distancia recorrida en km hasta el lugar
        """
        
        incremento_cansancio = tiempo_visita * 0.5 + distancia_km * 0.2  

        
        variacion = random.normalvariate(0, 0.25)  

        self.cansancio_acumulado += incremento_cansancio + variacion
        self.cansancio_acumulado = min(10, self.cansancio_acumulado)

        
        if tiempo_visita > 0.5 and random.random() < 0.7:  
            recuperacion = random.uniform(0.8, 2.0)  
            self.cansancio_acumulado = max(0, self.cansancio_acumulado - recuperacion)

    def _calcular_interes_lugar(self, lugar: Dict, preferencias_cliente: List[str] = None) -> float:
        """
        Calcula el interés del turista en un lugar específico basado en las preferencias del cliente

        Args:
            lugar: Diccionario con información del lugar
            preferencias_cliente: Lista de intereses/preferencias del cliente

        Returns:
            Nivel de interés (0-10)
        """
        tipo_lugar = lugar.get("tipo", "otro").lower()

        
        tipo_a_interes = {
            "museo": ["cultura", "historia", "arte", "museums"],
            "monumento": ["cultura", "historia", "arquitectura", "monuments"],
            "restaurante": ["gastronomía", "comida", "restaurants", "food"],
            "parque": ["naturaleza", "aire libre", "relax", "parks", "nature"],
            "playa": ["playa", "mar", "sol", "beaches", "relax"],
            "centro_comercial": ["compras", "shopping"],
            "teatro": ["cultura", "arte", "espectáculos", "entertainment"],
            "zoo": ["animales", "naturaleza", "familia", "nature"],
            "desplazamiento": ["transporte", "viaje"]
        }

        
        if not preferencias_cliente:
            interes_base = 7.0  
        else:
            
            categorias_lugar = tipo_a_interes.get(tipo_lugar, [])
            coincidencias = 0

            
            prefs_lower = [pref.lower() for pref in preferencias_cliente]

            
            for categoria in categorias_lugar:
                for pref in prefs_lower:
                    if categoria in pref or pref in categoria:
                        coincidencias += 1
                        break

            
            if coincidencias > 0:
                interes_base = min(9.2, 8.0 + coincidencias * 0.7)  
            else:
                interes_base = 6.0  

        
        popularidad = lugar.get("popularidad", 5)
        modificador_popularidad = (popularidad - 5) * 0.25  

        
        variacion = random.normalvariate(0.1, 0.7)  

        interes = interes_base + modificador_popularidad + variacion
        return max(4.5, min(10, interes))  

    def simular_visita(self, lugar: Dict, contexto: Dict) -> Dict:
        """
        Simula la visita a un lugar turístico

        Args:
            lugar: Diccionario con información del lugar visitado
            contexto: Condiciones externas (clima, día, hora, etc.)

        Returns:
            Diccionario con resultados de la simulación
        """
        
        clima_desc, valor_clima = self._generar_clima(
            contexto.get("temporada", "verano"),
            contexto.get("prob_lluvia", 0.18)  
        )

        crowding_desc, valor_crowding = self._generar_crowding(
            lugar,
            contexto.get("dia_semana", "sabado"),
            contexto.get("hora", 14)
        )

        atencion_desc, valor_atencion = self._generar_atencion(lugar.get("tipo", "otro"))

        tiempo_espera_desc, minutos_espera = self._generar_tiempo_espera(lugar, valor_crowding)

        
        preferencias_cliente = contexto.get("preferencias_cliente", [])
        interes = self._calcular_interes_lugar(lugar, preferencias_cliente)

        
        try:
            self.simulation.input['clima'] = valor_clima
            self.simulation.input['crowding'] = valor_crowding
            self.simulation.input['atencion'] = valor_atencion
            self.simulation.input['tiempo_espera'] = minutos_espera
            self.simulation.input['interes_lugar'] = interes

            self.simulation.compute()

            
            if hasattr(self.simulation.output, 'get'):
                satisfaccion_lugar = self.simulation.output.get('satisfaccion', 6.0)  
            else:
                satisfaccion_lugar = self.simulation.output['satisfaccion']

        except Exception as e:
            print(f"  Error en sistema fuzzy: {e}")
            print(f"     Usando satisfaccion por defecto basada en interes")
            
            satisfaccion_lugar = (
                    interes * 0.4 +  
                    valor_clima * 0.22 +  
                    valor_atencion * 0.22 +
                    (10 - valor_crowding) * 0.08 +  
                    (10 - min(minutos_espera/12, 10)) * 0.08  
            )
            
            satisfaccion_lugar = satisfaccion_lugar + 1.0  
            satisfaccion_lugar = max(4.5, min(10, satisfaccion_lugar))  

        
        tiempo_visita = self._generar_tiempo_visita(lugar, interes)

        
        self._actualizar_cansancio(tiempo_visita, contexto.get("distancia_km", 0))

        
        factor_cansancio = max(0.88, 1 - (self.cansancio_acumulado / 22))  
        satisfaccion_ajustada = satisfaccion_lugar * factor_cansancio

        
        satisfaccion_ajustada = satisfaccion_ajustada + 0.4  
        satisfaccion_ajustada = max(5.0, min(10, satisfaccion_ajustada))  

        
        if self.lugares_visitados:
            peso_visita_actual = 1 / (len(self.lugares_visitados) + 1)
            self.satisfaccion_general = (self.satisfaccion_general * (1 - peso_visita_actual) +
                                         satisfaccion_ajustada * peso_visita_actual)
        else:
            self.satisfaccion_general = satisfaccion_ajustada

        
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
            "satisfaccion": round(satisfaccion_ajustada, 1),
            "cansancio": round(self.cansancio_acumulado, 1)
        }

        self.lugares_visitados.append(visita)

        
        if satisfaccion_ajustada > 8:
            visita["comentario"] = f"¡Excelente experiencia en {lugar.get('nombre')}! {self._generar_comentario_positivo(lugar.get('tipo', 'lugar'), interes)}"
        elif satisfaccion_ajustada > 6:
            visita["comentario"] = f"Buena visita a {lugar.get('nombre')}. {self._generar_comentario_neutro(lugar.get('tipo', 'lugar'), valor_crowding)}"
        else:
            visita["comentario"] = f"Experiencia decepcionante en {lugar.get('nombre')}. {self._generar_comentario_negativo(lugar.get('tipo', 'lugar'), minutos_espera, valor_clima)}"

        return visita

    def _generar_comentario_positivo(self, tipo_lugar, interes) -> str:
        """Genera un comentario positivo según el tipo de lugar y nivel de interés"""
        comentarios = [
            "Superó mis expectativas.",
            "Definitivamente volvería.",
            "Una visita obligada.",
            "Valió totalmente la pena.",
            "Lo recomendaría sin dudas."
        ]

        if tipo_lugar == "restaurante":
            extras = ["La comida estaba deliciosa.", "Excelente servicio y ambiente."]
        elif tipo_lugar == "museo":
            extras = ["Las exhibiciones son fascinantes.", "El recorrido fue muy educativo."]
        elif tipo_lugar == "parque":
            extras = ["Un espacio muy relajante.", "Perfecto para desconectar."]
        elif tipo_lugar == "playa":
            extras = ["El agua estaba perfecta.", "Arena limpia y hermoso paisaje."]
        else:
            extras = ["Una experiencia memorable.", "Muy bien organizado."]

        return f"{random.choice(comentarios)} {random.choice(extras)}"

    def _generar_comentario_neutro(self, tipo_lugar, crowding) -> str:
        """Genera un comentario neutro según el tipo de lugar y nivel de crowding"""
        if crowding > 7:
            base = "Había demasiada gente, lo que le restó un poco a la experiencia."
        else:
            base = "Fue una experiencia correcta, aunque nada extraordinaria."

        if tipo_lugar == "restaurante":
            extras = ["La comida estaba bien.", "El servicio fue adecuado."]
        elif tipo_lugar == "museo":
            extras = ["Algunas exposiciones interesantes.", "Un recorrido estándar."]
        elif tipo_lugar == "parque":
            extras = ["Un lugar agradable para descansar.", "Está bien mantenido."]
        else:
            extras = ["Cumplió con lo básico.", "No me arrepiento, pero tampoco me sorprendió."]

        return f"{base} {random.choice(extras)}"

    def _generar_comentario_negativo(self, tipo_lugar, tiempo_espera, clima) -> str:
        """Genera un comentario negativo según el tipo de lugar y factores negativos"""
        if tiempo_espera > 45:
            base = f"La espera de {int(tiempo_espera)} minutos fue excesiva."
        elif clima < 4:
            base = "El mal clima arruinó gran parte de la experiencia."
        else:
            base = "No cumplió con mis expectativas."

        if tipo_lugar == "restaurante":
            extras = ["La comida dejó mucho que desear.", "La relación calidad-precio es mala."]
        elif tipo_lugar == "museo":
            extras = ["Exposiciones poco interesantes.", "Falta información en los recorridos."]
        elif tipo_lugar == "parque":
            extras = ["Mal mantenido.", "No hay suficientes facilidades."]
        else:
            extras = ["No lo recomendaría.", "Hay mejores opciones disponibles."]

        return f"{base} {random.choice(extras)}"

    def simular_itinerario(self, itinerario: List[Dict], contexto_base: Dict) -> Dict:
        """
        Simula la experiencia completa en un itinerario turístico considerando múltiples días

        Args:
            itinerario: Lista de lugares a visitar con sus características
            contexto_base: Condiciones generales del viaje (temporada, etc.)

        Returns:
            Resultados de la simulación
        """
        
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 7.5  
        self.lugares_visitados = []

        
        dia_actual = 1
        hora_actual = contexto_base.get("hora_inicio", 9)  
        lugares_por_dia = {}

        for i, lugar in enumerate(itinerario):
            
            dia_lugar = lugar.get("dia", dia_actual)

            
            if dia_lugar > dia_actual:
                
                self._aplicar_descanso_nocturno()

                
                dia_actual = dia_lugar
                hora_actual = contexto_base.get("hora_inicio", 9)

                
                if dia_actual not in lugares_por_dia:
                    lugares_por_dia[dia_actual] = []

            
            contexto_lugar = contexto_base.copy()
            contexto_lugar["hora"] = hora_actual
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

            
            if i < len(itinerario) - 1:
                
                siguiente_dia = itinerario[i+1].get("dia", dia_actual)
                if siguiente_dia == dia_actual:
                    tiempo_desplazamiento = contexto_lugar["distancia_km"] / 32  
                    hora_actual += tiempo_desplazamiento

        
        duracion_total_hrs = 0
        for dia, lugares in lugares_por_dia.items():
            
            duracion_total_hrs += min(12, len(lugares) * 2.5)  

        
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

    def _aplicar_descanso_nocturno(self):
        """
        Aplica el efecto del descanso nocturno sobre el cansancio acumulado
        """
        
        recuperacion_base = 6.5  

        
        factor_recuperacion = {
            "exigente": 0.85,  
            "relajado": 1.3,   
            "average": 1.1     
        }.get(self.tourist_profile, 1.1)

        
        recuperacion_total = recuperacion_base * factor_recuperacion * random.uniform(0.9, 1.3)  

        
        self.cansancio_acumulado = max(0, self.cansancio_acumulado - recuperacion_total)

        
        boost_satisfaccion = random.uniform(0.3, 0.6)  
        self.satisfaccion_general = min(10, self.satisfaccion_general + boost_satisfaccion)

    def _generar_valoracion_final(self) -> str:
        """Genera una valoración cualitativa de la experiencia completa"""
        if self.satisfaccion_general >= 8:
            return f"¡Experiencia excepcional! Con una satisfacción de {self.satisfaccion_general}/10, el viaje superó las expectativas. El itinerario fue excelente con una buena combinación de actividades."
        elif self.satisfaccion_general >= 6.5:
            return f"Viaje satisfactorio. Con una puntuación de {self.satisfaccion_general}/10, el itinerario funcionó bien aunque hubo algunos aspectos mejorables."
        elif self.satisfaccion_general >= 5:
            return f"Experiencia aceptable. La satisfacción general de {self.satisfaccion_general}/10 indica que el itinerario fue adecuado pero con varios puntos a mejorar."
        else:
            return f"Experiencia decepcionante. Con una puntuación de {self.satisfaccion_general}/10, este itinerario necesita una revisión completa para ajustarse mejor a las expectativas."

    def visualizar_resultados(self, resultados: Dict):
        """
        Visualiza los resultados de la simulación

        Args:
            resultados: Diccionario con resultados de la simulación
        """
        lugares = resultados["lugares_visitados"]
        nombres = [lugar["lugar"] for lugar in lugares]
        satisfacciones = [lugar["satisfaccion"] for lugar in lugares]

        plt.figure(figsize=(12, 8))

        
        plt.subplot(2, 1, 1)
        plt.bar(nombres, satisfacciones, color='skyblue')
        plt.axhline(y=resultados["satisfaccion_general"], color='r', linestyle='--', label='Satisfacción media')
        plt.xlabel('Lugar')
        plt.ylabel('Satisfacción (0-10)')
        plt.title(f'Satisfacción por lugar - Turista {resultados["perfil_turista"].capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        
        plt.subplot(2, 1, 2)
        cansancio = [lugar["cansancio"] for lugar in lugares]
        plt.plot(nombres, cansancio, marker='o', color='orange')
        plt.xlabel('Itinerario')
        plt.ylabel('Nivel de cansancio (0-10)')
        plt.title('Evolución del cansancio durante el itinerario')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig('simulacion_turista.png')
        print(f"Gráfico guardado como 'simulacion_turista.png'")
        plt.show()

    def receive(self, message: Dict, sender: Agent) -> Dict:
        """
        Procesa mensajes recibidos de otros agentes

        Args:
            message: Diccionario con el mensaje
            sender: Agente que envió el mensaje

        Returns:
            Respuesta en formato dict
        """
        try:
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

            elif msg_type == 'get_status':
                return {
                    'type': 'status',
                    'profile': self.tourist_profile,
                    'visits_count': len(self.lugares_visitados),
                    'current_satisfaction': self.satisfaccion_general
                }

            elif msg_type == 'change_profile':
                self.tourist_profile = message.get('profile', 'average')
                return {
                    'type': 'profile_changed',
                    'new_profile': self.tourist_profile
                }

            elif msg_type == 'simulate_single_place':
                
                place = message.get('place', {})
                context = message.get('context', {})
                result = self.simular_visita(place, context)
                return {
                    'type': 'single_simulation_result',
                    'result': result
                }

            return {
                'type': 'error',
                'msg': f"Tipo de mensaje no soportado: {msg_type}"
            }
        except Exception as e:
            return {
                'type': 'error',
                'msg': f"Error en {self.name}: {str(e)}"
            }