import { mcpHandler } from "../../../src/mcp.ts";

export const dynamic = "force-dynamic";

const handle = (req: Request) => mcpHandler(req);

export { handle as DELETE, handle as GET, handle as POST };
