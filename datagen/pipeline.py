#!/usr/bin/env python3
"""
Simplified video analysis pipeline coordinator that manages individual stage modules.
"""

from pathlib import Path
from typing import Dict, Any, List

import aiohttp

from stages.analysis import AnalysisStage
from stages.question_types import QuestionTypesStage
from stages.questions import QuestionStage
from stages.answers import AnswerStage
from stages.criteria import CriteriaStage
from stages.finalization import FinalizationStage


class VideoPipeline:
    """Pipeline coordinator that manages individual processing stages."""
    
    def __init__(self, session: aiohttp.ClientSession, base_url: str, model_name: str, max_concurrent: int, config: Dict[str, Any]):
        self.session = session
        self.base_url = base_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.config = config
        
        # Initialize all stages
        stage_args = (session, base_url, model_name, max_concurrent, config)
        self.analysis_stage = AnalysisStage(*stage_args)
        self.question_types_stage = QuestionTypesStage(*stage_args)
        self.question_stage = QuestionStage(*stage_args)
        self.answer_stage = AnswerStage(*stage_args)
        self.criteria_stage = CriteriaStage(*stage_args)
        self.finalization_stage = FinalizationStage(*stage_args)
    
    async def process_analysis(self, video_files: List[Path]) -> None:
        """Process video analysis stage."""
        await self.analysis_stage.process(video_files)
    
    async def process_question_types(self) -> None:
        """Process question types generation stage."""
        await self.question_types_stage.process()
    
    async def process_questions(self) -> None:
        """Process question generation stage."""
        await self.question_stage.process()
    
    async def process_answers(self) -> None:
        """Process answer generation stage."""
        await self.answer_stage.process()
    
    async def process_criteria(self) -> None:
        """Process criteria generation stage."""
        await self.criteria_stage.process()
    
    async def finalize_dataset(self) -> None:
        """Finalize dataset into individual samples."""
        await self.finalization_stage.process()
