"""
Utilidades para integrar Mistral AI API en el sistema
Reemplaza la funcionalidad de Google Generative AI (Gemini)
"""
from typing import Dict, Any, List, Optional
import os
from mistralai import Mistral, UserMessage

# Configuración de la API de Mistral
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# Modelo a usar (opciones comunes: mistral-large-latest, mistral-medium-latest, open-mixtral-8x7b)
DEFAULT_MODEL = "mistral-large-latest"

class MistralWrapper:
    """
    Wrapper para Mistral API que simula la interfaz de Gemini
    para facilitar la migración del código
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Inicializa el cliente de Mistral API

        Args:
            model_name: Nombre del modelo de Mistral a utilizar
        """
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY no encontrada en variables de entorno")

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model_name = model_name

    def generate_content(self,
                         prompt: str,
                         stream: bool = False,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Any:
        messages = [UserMessage(content=prompt)]
        response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return MistralResponse(response)

    def generate_multiple_responses(self, prompt: str, n: int = 1) -> List[Any]:
        responses = []
        for _ in range(n):
            response = self.generate_content(prompt)
            responses.append(response)
        return responses

class MistralResponse:
    def __init__(self, mistral_response: Any):
        self._response = mistral_response
        self.text = mistral_response.choices[0].message.content

    @property
    def response(self) -> Any:
        return self._response

def configure(api_key: str = None):
    global MISTRAL_API_KEY
    if api_key:
        MISTRAL_API_KEY = api_key
        os.environ["MISTRAL_API_KEY"] = api_key
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY no configurada. Usa configure(api_key='tu_api_key')")

class GenerativeModel:
    def __init__(self, model_name: str):
        self.mistral = MistralWrapper(model_name)
    def generate_content(self, prompt: str, **kwargs) -> Any:
        return self.mistral.generate_content(prompt, **kwargs)

