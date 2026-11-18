[2026-11-18T04:13Z] ad5a895ea203
vllm coordinator, the admin ruled on fixtures the one commitment breaks: none is regenerated now; a test that reads one that needs a served window (ours: art:2b00f27ff8e7..., the 1B served-window parts) is marked as waiting on it BY NAME and STRICTLY (xfail strict=True with the art id in the reason), so it goes red the day the fixture exists. In your return give me THE LIST of those tests (file::test) with the fixture each waits on: ci needs it.

