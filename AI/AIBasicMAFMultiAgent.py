import asyncio

from agent_framework import AgentResponseUpdate
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState
from azure.identity import DefaultAzureCredential


PROJECT_ENDPOINT = "https://foundryagentproject-resource.services.ai.azure.com/api/projects/foundryagentproject"
MODEL = "gpt-5.6-terra"
MAX_AGENT_TURNS = 4

VIDEO_TITLE = "AI-103 Study Cram"
VIDEO_DESCRIPTION = (
	"An overview video of key concepts covered in the AI-103 Microsoft AI exam "
	"which includes whiteboarding and demos."
)


async def main() -> None:
	chat_client = FoundryChatClient(
		project_endpoint=PROJECT_ENDPOINT,
		model=MODEL,
		credential=DefaultAzureCredential(),
	)

	creator = chat_client.as_agent(
		name="HashtagCreator",
		instructions=(
			"Create 5 to 8 relevant LinkedIn hashtags for the supplied video. "
			"Balance broad discovery with specific technical terms. On your first turn, "
			"propose a set. On later turns, revise it using the critic's feedback. "
			"Return only the hashtags."
		),
	)

	critic = chat_client.as_agent(
		name="HashtagCritic",
		instructions=(
			"Review only the most recent message from HashtagCreator against the original title "
			"and description; do not critique hashtags from older proposals. "
			"Identify weak, repetitive, or overly generic choices and give concise, actionable "
			"feedback. If the creator has already revised the hashtags, give a final verdict "
			"followed by the approved hashtag set."
		),
	)

	def round_robin_selector(state: GroupChatState) -> str:
		participant_names = list(state.participants.keys())
		return participant_names[state.current_round % len(participant_names)]

	workflow = GroupChatBuilder(
		participants=[creator, critic],
		selection_func=round_robin_selector,
		termination_condition=lambda messages: sum(
			1 for message in messages if message.role == "assistant"
		) >= MAX_AGENT_TURNS,
		intermediate_output_from=[creator, critic],
	).build()
	task = f"Video title: {VIDEO_TITLE}\nVideo description: {VIDEO_DESCRIPTION}"
	print(f"[User]\n{task}")

	last_speaker = None
	async for event in workflow.run(task, stream=True):
		if event.type == "intermediate" and isinstance(
			event.data, AgentResponseUpdate
		):
			speaker = event.data.author_name or "Assistant"
			if speaker != last_speaker:
				print(f"\n[{speaker}]")
				last_speaker = speaker
			print(event.data.text, end="", flush=True)

	print()


if __name__ == "__main__":
	asyncio.run(main())
