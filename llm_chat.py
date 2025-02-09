import os
import google.generativeai as genai

genai.configure(api_key='AIzaSyC00TQNE16s4A3R-CwMxW5CZiG-MULeUXo')

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

def initial_call(received_codes):
    system_prompt = """You are SwoleMate, an AI fitness assistant analyzing shoulder press form. You have received data from an AI model that uses 6 key points from MediaPipe (points 11, 13, 15 on the left and 12, 14, 16 on the right) to assess shoulder press form.

The metrics are scored from 0-6, where:
0: Perfect form
1: Left shoulder stability needs work
2: Left elbow angle needs adjustment (goal: 90°)
3: Left wrist alignment needs correction
4: Right shoulder position needs adjustment
5: Right elbow angle needs correction (goal: 90°)
6: Right wrist alignment needs work

Based on the scores provided ({received_codes}), provide a detailed analysis and personalized advice. Format your response as a JSON object with the following structure, and ONLY return the JSON object with no additional text:

{{
    "user_advice": {{
        "introduction": "A personalized introduction summarizing the analysis",
        "left_side": {{
            "shoulder": {{
                "status": "Good/Needs Improvement/Needs Adjustment",
                "issue_detected": "Description of the issue if any",
                "advice": "Specific advice for improvement",
                "exercises": [
                    {{
                        "name": "Exercise name",
                        "description": "Brief description",
                        "youtube_example": "URL or 'This is a general tip'"
                    }}
                ]
            }},
            "elbow": {{ ... }},
            "wrist": {{ ... }}
        }},
        "right_side": {{
            "shoulder": {{ ... }},
            "elbow": {{ ... }},
            "wrist": {{ ... }}
        }},
        "general_advice": {{
            "warm_up": "Warm-up advice",
            "weight_selection": "Weight selection guidance",
            "breathing": "Breathing technique",
            "core_engagement": "Core engagement tips",
            "consistency": "Consistency advice",
            "safety": "Safety considerations"
        }}
    }}
}}"""

    chat = model.start_chat(history=[])
    response = chat.send_message(system_prompt)
    return response.text

def chat_call(user_query, history):
    # Create the system prompt
    system_prompt = """You are SwoleMate, an AI fitness assistant specializing in shoulder press form analysis. 
    You help users improve their exercise technique by providing detailed, personalized advice. Format your response as a JSON object with the following structure. The response should only be JSON and nothing else:
    
    {
        "response": {
            "message": "Your detailed response here",
            "resources": [
                {
                    "type": "video/article/tip",
                    "title": "Resource title",
                    "description": "Brief description",
                    "url": "URL if applicable"
                }
            ],
            "key_points": [
                "Key point 1",
                "Key point 2",
                "etc..."
            ]
        }
    }

    When answering questions:
    1. Be clear and concise
    2. Use proper exercise terminology
    3. Provide specific, actionable advice
    4. Include relevant YouTube video links when appropriate
    5. Focus on safety and proper form
    6. Draw from your knowledge of shoulder press mechanics and common form issues"""
    
    # Format the conversation context from history
    context = ""
    if history:
        for entry in history:
            if entry.get('role') == 'user':
                context += f"User: {entry.get('message', '')}\n"
            elif entry.get('role') == 'assistant':
                context += f"Assistant: {entry.get('message', '')}\n"
    
    # Combine everything into a single prompt
    full_prompt = f"{system_prompt}\n\nPrevious conversation:\n{context}\nUser: {user_query}\nAssistant: Please provide your response in the specified JSON format."
    
    # Get response from the model
    response = model.generate_content(full_prompt)
    
    try:
        # Parse the JSON response
        import json
        # Remove markdown code block if present
        cleaned_response = response.text.replace('```json', '').replace('```', '').strip()
        response_json = json.loads(cleaned_response)
        
        # Format the response for display
        formatted_response = response_json['response']['message']
        # Convert markdown formatting to HTML
        formatted_response = formatted_response.replace('**', '')  # Remove bold markdown
        formatted_response = formatted_response.replace('\n\n', '<br><br>')  # Convert double newlines to HTML breaks
        
        # Add resources if they exist
        if response_json['response'].get('resources'):
            formatted_response += "\n\nResources:"
            for resource in response_json['response']['resources']:
                if resource.get('url'):
                    formatted_response += f"\n• <b>{resource['title']}</b>: {resource['description']}\n  <a href='{resource['url']}' target='_blank'>{resource['url']}</a>"
                else:
                    formatted_response += f"\n• <b>{resource['title']}</b>: {resource['description']}"
        
        # Add key points if they exist
        if response_json['response'].get('key_points'):
            formatted_response += "\n\nKey Points:"
            for point in response_json['response']['key_points']:
                formatted_response += f"\n• {point}"
        
        return formatted_response
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response.text}")
        return "I apologize, but I encountered an error formatting the response. Please try asking your question again."