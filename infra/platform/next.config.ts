import type { NextConfig } from "next";
import { withWorkflow } from "workflow/next";

const config: NextConfig = {
  serverExternalPackages: ["@electric-sql/pglite", "pg"],
};

export default withWorkflow(config);
