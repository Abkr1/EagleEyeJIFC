"""
Web routes serving HTML pages for the EagleEye dashboard.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pathlib import Path

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_page": "dashboard",
    })


@router.get("/incidents", response_class=HTMLResponse)
async def incidents(request: Request):
    return templates.TemplateResponse("incidents.html", {
        "request": request,
        "active_page": "incidents",
    })


@router.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "active_page": "analysis",
    })


@router.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request):
    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "active_page": "predictions",
    })


@router.get("/alerts", response_class=HTMLResponse)
async def alerts(request: Request):
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "active_page": "alerts",
    })


@router.get("/intel", response_class=HTMLResponse)
async def intel(request: Request):
    return templates.TemplateResponse("intel.html", {
        "request": request,
        "active_page": "intel",
    })


@router.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "active_page": "reports",
    })


@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "active_page": "settings",
    })
