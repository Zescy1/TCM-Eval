from opencompass.models import Ollama

models = [
    dict(
        type=Ollama,
        abbr='huatuo',             
        path='ollama',                
        model='huatuo',               
        api_base='http://localhost:11435/api/generate',  
        max_seq_len=9182,             
        max_out_len=9182,             
        batch_size=1,                 
        generation_kwargs=dict(
            temperature=0.01,                     
        ),
        query_per_second=5,      
    )
]

