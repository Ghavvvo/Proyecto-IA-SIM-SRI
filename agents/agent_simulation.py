import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Tuple, Any
from autogen import Agent

# Configure matplotlib before importing pyplot
import os
import sys
# Check if we're in experiment/testing mode
if any('experiment' in arg for arg in sys.argv) or os.environ.get('TESTING', '').lower() == 'true':
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

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

        # Configurar perfiles de turista (valores de 0 a 10)
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

        # Inicializar sistema de lógica difusa
        self._setup_fuzzy_system()

        # Variables de estado
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 7.0  # Empezamos con una satisfacción neutral-positiva
        self.lugares_visitados = []

    def _setup_fuzzy_system(self):
        """Configurar el sistema de lógica difusa para evaluación de experiencias"""
        # Definir variables de entrada (antecedentes)
        self.clima = ctrl.Antecedent(np.arange(0, 11, 1), 'clima')
        self.crowding = ctrl.Antecedent(np.arange(0, 11, 1), 'crowding')
        self.atencion = ctrl.Antecedent(np.arange(0, 11, 1), 'atencion')
        self.tiempo_espera = ctrl.Antecedent(np.arange(0, 121, 1), 'tiempo_espera')
        self.interes_lugar = ctrl.Antecedent(np.arange(0, 11, 1), 'interes_lugar')

        # Definir variable de salida (consecuente)
        self.satisfaccion = ctrl.Consequent(np.arange(0, 11, 1), 'satisfaccion')

        # Definir conjuntos difusos para cada variable
        # Clima (0: muy malo, 10: excelente)
        self.clima['malo'] = fuzz.trimf(self.clima.universe, [0, 0, 5])
        self.clima['moderado'] = fuzz.trimf(self.clima.universe, [2, 5, 8])
        self.clima['bueno'] = fuzz.trimf(self.clima.universe, [5, 10, 10])

        # Crowding (0: vacío, 10: extremadamente lleno)
        self.crowding['bajo'] = fuzz.trimf(self.crowding.universe, [0, 0, 4])
        self.crowding['medio'] = fuzz.trimf(self.crowding.universe, [3, 5, 7])
        self.crowding['alto'] = fuzz.trimf(self.crowding.universe, [6, 10, 10])

        # Atención (0: pésima, 10: excelente)
        self.atencion['mala'] = fuzz.trimf(self.atencion.universe, [0, 0, 4])
        self.atencion['regular'] = fuzz.trimf(self.atencion.universe, [3, 5, 7])
        self.atencion['buena'] = fuzz.trimf(self.atencion.universe, [6, 10, 10])

        # Tiempo de espera en minutos (0: sin espera, 120: espera excesiva)
        self.tiempo_espera['corto'] = fuzz.trimf(self.tiempo_espera.universe, [0, 0, 20])
        self.tiempo_espera['medio'] = fuzz.trimf(self.tiempo_espera.universe, [15, 30, 60])
        self.tiempo_espera['largo'] = fuzz.trimf(self.tiempo_espera.universe, [45, 120, 120])

        # Interés en el lugar (0: ningún interés, 10: máximo interés)
        self.interes_lugar['bajo'] = fuzz.trimf(self.interes_lugar.universe, [0, 0, 4])
        self.interes_lugar['medio'] = fuzz.trimf(self.interes_lugar.universe, [3, 5, 7])
        self.interes_lugar['alto'] = fuzz.trimf(self.interes_lugar.universe, [6, 10, 10])

        # Satisfacción (0: insatisfecho, 10: completamente satisfecho)
        self.satisfaccion['baja'] = fuzz.trimf(self.satisfaccion.universe, [0, 0, 4])
        self.satisfaccion['media'] = fuzz.trimf(self.satisfaccion.universe, [3, 5, 7])
        self.satisfaccion['alta'] = fuzz.trimf(self.satisfaccion.universe, [6, 10, 10])

        # Definir reglas
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

        # Crear sistema de control
        self.system = ctrl.ControlSystem([
            regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8
        ])

        # Crear simulación
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
        # Ajustar probabilidades según temporada
        if temporada == "verano":
            temp_base = 28
            temp_var = 5
            prob_lluvia = probabilidad_lluvia * 0.5
        elif temporada == "invierno":
            temp_base = 10
            temp_var = 8
            prob_lluvia = probabilidad_lluvia * 1.5
        elif temporada == "otoño":
            temp_base = 18
            temp_var = 7
            prob_lluvia = probabilidad_lluvia * 1.2
        else:  # primavera
            temp_base = 22
            temp_var = 6
            prob_lluvia = probabilidad_lluvia

        # Generar temperatura
        temperatura = round(random.normalvariate(temp_base, temp_var/3), 1)

        # Determinar si llueve
        esta_lloviendo = random.random() < prob_lluvia

        # Valoración climática para el sistema difuso (0-10)
        if esta_lloviendo:
            valor_clima = max(0, 10 - random.randint(4, 8))  # La lluvia reduce significativamente la valoración
            descripcion = f"Lluvia, {temperatura}°C"
        else:
            # Calcular valor climático según la temperatura y preferencia de temporada
            if temporada == "verano":
                if temperatura < 18:
                    valor_clima = 5  # Frío en verano no es ideal
                elif temperatura > 35:
                    valor_clima = 4  # Demasiado calor tampoco es bueno
                else:
                    valor_clima = min(10, max(5, 10 - abs(temperatura - 26)))
            elif temporada == "invierno":
                if temperatura < 0:
                    valor_clima = 4  # Demasiado frío
                else:
                    valor_clima = min(10, max(5, 10 - abs(temperatura - 12)))
            else:  # primavera/otoño
                valor_clima = min(10, max(5, 10 - abs(temperatura - 22)))

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
        # Factor base de popularidad del lugar (0-10)
        popularidad_base = lugar.get("popularidad", 5)

        # Modificadores por día de semana
        modificador_dia = {
            "lunes": 0.6,
            "martes": 0.6,
            "miercoles": 0.7,
            "jueves": 0.8,
            "viernes": 1.0,
            "sabado": 1.5,
            "domingo": 1.3
        }.get(dia_semana.lower(), 1.0)

        # Modificadores por hora
        if hora < 9 or hora > 20:
            modificador_hora = 0.4
        elif 11 <= hora <= 14 or 17 <= hora <= 19:  # Horas pico
            modificador_hora = 1.5
        else:
            modificador_hora = 1.0

        # Calcular nivel de crowding con componente aleatorio
        nivel_base = popularidad_base * modificador_dia * modificador_hora
        variacion = random.uniform(-1.5, 1.5)
        nivel_crowding = max(0, min(10, nivel_base + variacion))

        # Descripción cualitativa
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
        # Niveles base según tipo de lugar
        niveles_base = {
            "museo": 7.5,
            "restaurante": 6.5,
            "parque": 6.0,
            "monumento": 6.5,
            "hotel": 7.5,
            "playa": 5.5,
            "centro_comercial": 6.0,
            "teatro": 7.0,
            "zoo": 6.5
        }

        nivel_base = niveles_base.get(tipo_lugar.lower(), 6.0)

        # Añadir variación aleatoria
        variacion = random.normalvariate(0, 1.5)
        nivel_atencion = max(0, min(10, nivel_base + variacion))

        # Descripción cualitativa
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

        # Tiempos base según tipo de lugar (en minutos)
        tiempos_base = {
            "museo": 15,
            "restaurante": 20,
            "parque": 5,
            "monumento": 10,
            "hotel": 8,
            "playa": 2,
            "centro_comercial": 5,
            "teatro": 12,
            "zoo": 15
        }

        tiempo_base = tiempos_base.get(tipo_lugar.lower(), 10)

        # El crowding impacta exponencialmente en el tiempo de espera
        factor_crowding = (nivel_crowding / 5) ** 2

        # Calcular tiempo con variación aleatoria
        tiempo_espera = tiempo_base * factor_crowding * random.uniform(0.7, 1.3)
        tiempo_espera = max(0, min(120, tiempo_espera))  # Límite máximo de 120 minutos

        # Descripción cualitativa
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

        # Tiempos base según tipo de lugar (en horas)
        tiempos_base = {
            "museo": 1.5,
            "restaurante": 1.2,
            "parque": 1.0,
            "monumento": 0.5,
            "hotel": 0.2,  # Tiempo para check-in/check-out
            "playa": 2.0,
            "centro_comercial": 1.5,
            "teatro": 2.0,
            "zoo": 2.5
        }

        tiempo_base = tiempos_base.get(tipo_lugar.lower(), 1.0)

        # El interés modifica el tiempo de visita
        factor_interes = 0.6 + (nivel_interes / 10) * 0.8

        # Variación aleatoria
        variacion = random.normalvariate(0, 0.2)

        tiempo_visita = tiempo_base * factor_interes + variacion
        return max(0.1, tiempo_visita)  # Mínimo 6 minutos de visita

    def _actualizar_cansancio(self, tiempo_visita: float, distancia_km: float):
        """
        Actualiza el nivel de cansancio acumulado

        Args:
            tiempo_visita: Tiempo de visita en horas
            distancia_km: Distancia recorrida en km hasta el lugar
        """
        # El cansancio aumenta con el tiempo y la distancia
        incremento_cansancio = tiempo_visita * 0.8 + distancia_km * 0.3

        # Añadir aleatoriedad al cansancio
        variacion = random.normalvariate(0, 0.5)

        self.cansancio_acumulado += incremento_cansancio + variacion
        self.cansancio_acumulado = min(10, self.cansancio_acumulado)

        # Si el turista descansa (ej: restaurante, hotel), puede recuperarse
        if tiempo_visita > 1 and random.random() < 0.7:
            recuperacion = random.uniform(0.5, 2.0)
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
        
        # Mapeo de tipos de lugares a categorías de interés
        tipo_a_interes = {
            "museo": ["cultura", "historia", "arte", "museums"],
            "monumento": ["cultura", "historia", "arquitectura", "monuments"],
            "restaurante": ["gastronomía", "comida", "restaurants", "food"],
            "parque": ["naturaleza", "aire libre", "relax", "parks", "nature"],
            "playa": ["playa", "mar", "sol", "beaches", "relax"],
            "centro_comercial": ["compras", "shopping"],
            "teatro": ["cultura", "arte", "espectáculos", "entertainment"],
            "zoo": ["animales", "naturaleza", "familia", "nature"]
        }
        
        # Si no hay preferencias, usar un interés base moderado
        if not preferencias_cliente:
            interes_base = 6.0
        else:
            # Calcular interés basado en coincidencia con preferencias
            categorias_lugar = tipo_a_interes.get(tipo_lugar, [])
            coincidencias = 0
            
            # Convertir preferencias a minúsculas para comparación
            prefs_lower = [pref.lower() for pref in preferencias_cliente]
            
            # Contar coincidencias
            for categoria in categorias_lugar:
                for pref in prefs_lower:
                    if categoria in pref or pref in categoria:
                        coincidencias += 1
                        break
            
            # Calcular interés base
            if coincidencias > 0:
                interes_base = min(9, 7 + coincidencias * 1.5)  # Alto interés si coincide
            else:
                interes_base = 4  # Bajo interés si no coincide con preferencias
        
        # Modificar según popularidad del lugar
        popularidad = lugar.get("popularidad", 5)
        modificador_popularidad = (popularidad - 5) * 0.2
        
        # Variación aleatoria menor (no depende del perfil)
        variacion = random.normalvariate(0, 0.8)
        
        interes = interes_base + modificador_popularidad + variacion
        return max(0, min(10, interes))

    def simular_visita(self, lugar: Dict, contexto: Dict) -> Dict:
        """
        Simula la visita a un lugar turístico

        Args:
            lugar: Diccionario con información del lugar visitado
            contexto: Condiciones externas (clima, día, hora, etc.)

        Returns:
            Diccionario con resultados de la simulación
        """
        # Obtener factores aleatorios
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

        # Calcular interés en el lugar usando las preferencias del cliente
        preferencias_cliente = contexto.get("preferencias_cliente", [])
        interes = self._calcular_interes_lugar(lugar, preferencias_cliente)

        # Ejecutar sistema de lógica difusa
        try:
            self.simulation.input['clima'] = valor_clima
            self.simulation.input['crowding'] = valor_crowding
            self.simulation.input['atencion'] = valor_atencion
            self.simulation.input['tiempo_espera'] = minutos_espera
            self.simulation.input['interes_lugar'] = interes
            
            # Debug: imprimir valores de entrada (sin emojis para evitar problemas de encoding)
            if __debug__:  # Solo en modo debug
                print(f"  Valores fuzzy - Clima: {valor_clima}, Crowding: {valor_crowding}, Atencion: {valor_atencion}")
                print(f"     Tiempo espera: {minutos_espera} min, Interes: {interes}")
            
            self.simulation.compute()
            
            # Verificar si hay salida válida
            if hasattr(self.simulation.output, 'get'):
                satisfaccion_lugar = self.simulation.output.get('satisfaccion', 5.0)
            else:
                satisfaccion_lugar = self.simulation.output['satisfaccion']
                
        except Exception as e:
            print(f"  Error en sistema fuzzy: {e}")
            print(f"     Usando satisfaccion por defecto basada en interes")
            # Fallback: usar una fórmula simple basada en los inputs
            satisfaccion_lugar = (
                interes * 0.4 +  # El interés es el factor más importante
                valor_clima * 0.2 +
                valor_atencion * 0.2 +
                (10 - valor_crowding) * 0.1 +  # Menos crowding es mejor
                (10 - min(minutos_espera/12, 10)) * 0.1  # Menos espera es mejor
            )
            satisfaccion_lugar = max(0, min(10, satisfaccion_lugar))


        # Calcular tiempo de visita
        tiempo_visita = self._generar_tiempo_visita(lugar, interes)

        # Actualizar cansancio
        self._actualizar_cansancio(tiempo_visita, contexto.get("distancia_km", 0))

        # Aplicar factor de cansancio a la satisfacción
        factor_cansancio = max(0.7, 1 - (self.cansancio_acumulado / 15))
        satisfaccion_ajustada = satisfaccion_lugar * factor_cansancio

        # Actualizar satisfacción general (promedio ponderado)
        if self.lugares_visitados:
            peso_visita_actual = 1 / (len(self.lugares_visitados) + 1)
            self.satisfaccion_general = (self.satisfaccion_general * (1 - peso_visita_actual) +
                                        satisfaccion_ajustada * peso_visita_actual)
        else:
            self.satisfaccion_general = satisfaccion_ajustada

        # Registrar visita
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

        # Añadir comentarios cualitativos basados en la satisfacción
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
        # Reiniciar estado
        self.cansancio_acumulado = 0
        self.satisfaccion_general = 7.0
        self.lugares_visitados = []
        
        # Variables para tracking de días
        dia_actual = 1
        hora_actual = contexto_base.get("hora_inicio", 9)  # Hora de inicio predeterminada: 9 AM
        lugares_por_dia = {}
        
        for i, lugar in enumerate(itinerario):
            # Detectar cambio de día basado en el campo 'dia' del lugar
            dia_lugar = lugar.get("dia", dia_actual)
            
            # Si cambiamos de día, aplicar descanso nocturno
            if dia_lugar > dia_actual:
                # Aplicar descanso nocturno
                self._aplicar_descanso_nocturno()
                
                # Actualizar día y resetear hora
                dia_actual = dia_lugar
                hora_actual = contexto_base.get("hora_inicio", 9)
                
                # Registrar cambio de día en los resultados
                if dia_actual not in lugares_por_dia:
                    lugares_por_dia[dia_actual] = []
            
            # Actualizar contexto específico para este lugar
            contexto_lugar = contexto_base.copy()
            contexto_lugar["hora"] = hora_actual
            contexto_lugar["dia_actual"] = dia_actual

            # Calcular distancia desde lugar anterior o punto de partida
            if i > 0 and lugar.get("dia", dia_actual) == itinerario[i-1].get("dia", dia_actual):
                # Si es el mismo día, usar distancia desde lugar anterior
                contexto_lugar["distancia_km"] = lugar.get("distancia_anterior", 2)
            else:
                # Si es un nuevo día, usar distancia desde punto de inicio (hotel)
                contexto_lugar["distancia_km"] = lugar.get("distancia_inicio", 5)

            # Simular visita
            resultado_visita = self.simular_visita(lugar, contexto_lugar)
            resultado_visita["dia"] = dia_actual  # Añadir información del día
            
            # Registrar en qué día se visitó cada lugar
            if dia_actual not in lugares_por_dia:
                lugares_por_dia[dia_actual] = []
            lugares_por_dia[dia_actual].append(resultado_visita["lugar"])

            # Actualizar hora para próximo lugar
            tiempo_total = resultado_visita["tiempo_espera_min"] / 60 + resultado_visita["tiempo_visita_hrs"]
            hora_actual += tiempo_total

            # Añadir tiempo de desplazamiento al siguiente lugar
            if i < len(itinerario) - 1:
                # Solo añadir desplazamiento si el siguiente lugar es del mismo día
                siguiente_dia = itinerario[i+1].get("dia", dia_actual)
                if siguiente_dia == dia_actual:
                    tiempo_desplazamiento = contexto_lugar["distancia_km"] / 30  # 30 km/h en promedio
                    hora_actual += tiempo_desplazamiento

        # Calcular duración total considerando múltiples días
        duracion_total_hrs = 0
        for dia, lugares in lugares_por_dia.items():
            # Sumar horas activas por día (asumiendo 12 horas activas por día)
            duracion_total_hrs += min(12, len(lugares) * 2.5)  # Estimación aproximada

        # Resultados globales
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
        # Recuperación base por dormir
        recuperacion_base = 5.0
        
        # Factor de recuperación según el perfil
        factor_recuperacion = {
            "exigente": 0.7,  # Los exigentes descansan menos efectivamente
            "relajado": 1.2,  # Los relajados descansan mejor
            "average": 1.0
        }.get(self.tourist_profile, 1.0)
        
        # Aplicar recuperación con algo de aleatoriedad
        recuperacion_total = recuperacion_base * factor_recuperacion * random.uniform(0.8, 1.2)
        
        # Reducir el cansancio acumulado
        self.cansancio_acumulado = max(0, self.cansancio_acumulado - recuperacion_total)
        
        # Pequeño boost a la satisfacción general por empezar un nuevo día
        boost_satisfaccion = random.uniform(0.1, 0.3)
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

        # Gráfico de satisfacción por lugar
        plt.subplot(2, 1, 1)
        plt.bar(nombres, satisfacciones, color='skyblue')
        plt.axhline(y=resultados["satisfaccion_general"], color='r', linestyle='--', label='Satisfacción media')
        plt.xlabel('Lugar')
        plt.ylabel('Satisfacción (0-10)')
        plt.title(f'Satisfacción por lugar - Turista {resultados["perfil_turista"].capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Evolución del cansancio
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
                # Cambiar perfil si se especifica
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
                # Simular una sola visita
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

