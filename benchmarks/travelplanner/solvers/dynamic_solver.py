"""
Dynamic OR-Tools Solver Generation using LLM

LLM으로 제약 조건에 맞는 OR-Tools CP-SAT 솔버 코드를 동적 생성
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def generate_solver_code_from_constraints(
    constraints: Dict[str, Any],
    model: str = "gpt-4o"
) -> str:
    """
    LLM을 사용하여 제약 조건으로부터 OR-Tools 솔버 코드 생성
    
    Args:
        constraints: 제약 조건 딕셔너리
        model: 사용할 LLM 모델
    
    Returns:
        실행 가능한 Python 솔버 함수 코드 (문자열)
    """
    llm = ChatOpenAI(model=model, temperature=0)
    
    constraints_str = json.dumps(constraints, indent=2, ensure_ascii=False)
    
    prompt = f"""다음 제약 조건으로 OR-Tools CP-SAT 여행 계획 솔버를 생성하세요.

제약 조건:
{constraints_str}

요구사항:
1. 함수명: solve_travel_plan(candidates, constraints)
2. candidates는 flights, hotels, restaurants, attractions 포함
3. CP-SAT 모델로 의사결정 변수 생성
4. 제약 추가:
   - 예산: total_cost <= budget
   - 시간: daily_time <= max_daily_activity_time
   - 선택 규칙: 출발/귀국 항공편 각 1개, 호텔 1개, 하루 3끼, 하루 2-4개 관광지
   - 카테고리 필터링: forbidden_categories 제외
   - 숙소 규칙: house_rule 일치하는 호텔만
   - 교통수단: transportation_mode 제약 적용
5. 목적 함수: maximize (100 * preference_score - 1 * total_cost)
6. 반환: {{"status": "optimal", "total_cost": float, "flights": {{}}, "hotel": {{}}, "daily_plans": []}}

함수 정의만 출력하세요. 설명 없이 실행 가능한 Python 코드만:
```python
def solve_travel_plan(candidates, constraints):
```

import 문은 함수 안에 포함하세요."""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    code = response.content.strip()
    
    # 코드 블록 추출
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    return code


def execute_dynamic_solver(
    solver_code: str,
    candidates: Dict[str, Any],
    constraints: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    동적 생성된 솔버 코드를 메모리에서 실행
    
    Args:
        solver_code: 생성된 솔버 함수 코드
        candidates: 후보 항목 (flights, hotels 등)
        constraints: 제약 조건
    
    Returns:
        솔버 실행 결과 또는 None (실패 시)
    """
    local_vars = {}
    
    try:
        # 코드 실행하여 함수 정의
        exec(solver_code, local_vars)
        
        # solve_travel_plan 함수 가져오기
        solve_fn = local_vars.get('solve_travel_plan')
        
        if not solve_fn:
            print("[오류] solve_travel_plan 함수를 찾을 수 없습니다")
            return None
        
        # 솔버 실행
        result = solve_fn(candidates, constraints)
        return result
        
    except Exception as e:
        print(f"[오류] 동적 솔버 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
