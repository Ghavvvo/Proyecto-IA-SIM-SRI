from typing import List
import google.generativeai as genai

# Configuración de la API de Gemini
GEMINI_API_KEY = "AIzaSyDmW-QXAeksN6hacpCMVpTQnOEAD8MLG00"
genai.configure(api_key=GEMINI_API_KEY)

class RAGSystem:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.answersContext = []
        # Crear instancia del modelo generativo
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """
        Recupera los fragmentos más relevantes para la consulta del usuario.
        Args:
            query (str): Consulta del usuario.
            top_k (int): Número de fragmentos relevantes a recuperar.
        Returns:
            List[str]: Lista de textos relevantes.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return [doc for doc in results['documents'][0]]

    def generate(self, query: str, context: List[str]) -> str:
        """
        Genera una respuesta basada en la consulta y el contexto recuperado.
        Args:
            query (str): Consulta del usuario.
            context (List[str]): Fragmentos relevantes recuperados.
        Returns:
            str: Respuesta generada por el modelo.
        """

        prompt = f"Hola, necesito que te comportes como un guía turístico experto, estoy planificando mis vacaciones y necesito que me ayudes con lo siguiente: {query}\n\n. Dame una lista de 5 de mis mejores opciones teniendo en cuenta que estos son los mejores y únicos lugares que puedo visitar: \n{context}.\n Sé conciso, y necesito que siempre me des opciones, aunque no sean las mas adecuadas, no me digas más que el listado de opciones y una breve introducción"

        try:
            response = self.model.generate_content(prompt)
            self.answersContext.insert(0, f'Yo: {query}, \nAsistente de turismo: {response.text}\n')
            return response.text
        except AttributeError as e:
            print(f"Error al generar contenido: {e}")
            return "No se pudo generar una respuesta."

    def rag_query(self, query: str) -> str:
        """
        Implementa el flujo completo de RAG: recuperación y generación.
        Args:
            query (str): Consulta del usuario.
        Returns:
            str: Respuesta generada.
        """
        self.answersContext.append(f'Yo: {query}, \n Asistente de turismo: No hay problema')
        prevContext = self.model.generate_content(f"""
                   Hola, te voy enviar una conversación previa que he tenido con un asistente de turismo, necesito que me resumas en máximo {4 * len(self.answersContext)} palabras lo que deseo, no te inventes cosas, sé conciso, y la respuesta proporcionala en primera persona:
                   \n{self.answersContext}
               """)

        context = self.retrieve(prevContext.text)
        return self.generate(prevContext.text, context)
