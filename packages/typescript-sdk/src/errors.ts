export class KothaGPTError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "KothaGPTError";
  }
}

export class APIError extends KothaGPTError {
  constructor(
    public status: number,
    message: string,
    public body?: unknown,
  ) {
    super(message);
    this.name = "APIError";
  }
}

export class AuthenticationError extends APIError {
  constructor(status: number, message: string, body?: unknown) {
    super(status, message, body);
    this.name = "AuthenticationError";
  }
}

export class NotFoundError extends APIError {
  constructor(status: number, message: string, body?: unknown) {
    super(status, message, body);
    this.name = "NotFoundError";
  }
}

export async function checkStatus(response: Response): Promise<Response> {
  if (response.ok) return response;
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  const message =
    (body as { detail?: string } | undefined)?.detail ??
    `Request failed with status ${response.status}`;
  if (response.status === 401) throw new AuthenticationError(response.status, message, body);
  if (response.status === 404) throw new NotFoundError(response.status, message, body);
  throw new APIError(response.status, message, body);
}