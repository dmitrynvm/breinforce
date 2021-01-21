import uvicorn
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

'''
round_state = {
    'dealer_btn': 4,
    'big_blind_pos': 6,
    'straddle_pos': 0,
    'small_blind_amount': 1,
    'street': 'preflop',
    'community_card': [],
    'seats': [
        {
            'stack': 351,
            'state': 'participating',
            'name': '现实点好饿',
            'uuid': '现实点好饿'
        },
        {
            'stack': 228,
            'state': 'participating',
            'name': 'ME',
            'uuid': 'ME'
        }
    ],
    'pot': {
        'main':
            {
                'amount': 14
            }
    },
    'action_histories': {
        'preflop': [
            {
                'action': 'straddle',
                'amount': 4,
                'name': '现实点好饿',
                'add_amount': 4,
                'uuid': '现实点好饿'
            },
            {
                'action': 'call',
                'amount': 4,
                'name': '挑聘左',
                'add_amount': 0,
                'uuid': '挑聘左'
            },
        ]
    }
}
hole_card = ['HJ', 'S6']
valid_actions = [
    {
        'action': 'fold',
        'amount': 0
    },
    {
        'action': 'call',
        'amount': 4
    },
    {
        'action': 'raise',
        'amount': {
            'min': 7.2,
            'max': 228
        }
    }
]
'''


class Poker(BaseModel):
    """
    """
    data: dict

app = FastAPI()


@app.get("/")
def touch():
    return {"Breinforce": "works"}


@app.post("/")
def predict(data: Poker):
    print(data)
    return {"item_id": "q"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=True
    )
