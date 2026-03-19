import { NextResponse } from 'next/server';

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const { id } = params;
  const backendUrl = process.env.NEXT_PUBLIC_GPTR_API_URL || 'http://localhost:8000';

  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get('session_id');
    const endpoint = sessionId
      ? `${backendUrl}/api/reports/${id}/workflow?session_id=${encodeURIComponent(sessionId)}`
      : `${backendUrl}/api/reports/${id}/workflow`;

    const response = await fetch(endpoint);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Error ${response.status}` }));
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch workflow' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error(`GET /api/reports/${id}/workflow - Error proxying to backend:`, error);
    return NextResponse.json(
      { error: 'Failed to connect to backend service' },
      { status: 500 }
    );
  }
}
