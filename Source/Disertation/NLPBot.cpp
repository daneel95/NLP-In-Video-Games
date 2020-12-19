// Fill out your copyright notice in the Description page of Project Settings.


#include "NLPBot.h"


#include "DisertationAIController.h"
#include "Runtime/Online/HTTP/Public/Http.h"
#include "Runtime/online/http/Public/GenericPlatform/GenericPlatformHttp.h"
#include "Components/CapsuleComponent.h"
#include "Kismet/GameplayStatics.h"
#include "DisertationPlayerController.h"
#include "Location.h"
#include "Kismet/KismetMathLibrary.h"


// Sets default values
ANLPBot::ANLPBot()
{
	bIsBotThinking = false;
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);
	AIControllerClass = ADisertationAIController::StaticClass();
	bUseControllerRotationYaw = false;
}

// Called when the game starts or when spawned
void ANLPBot::BeginPlay()
{
	Super::BeginPlay();
	
	// Call the flask server to send all possible locations for further calls
	SendLocationsToServer();	
}

void ANLPBot::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	FRotator NewRotator = UKismetMathLibrary::MakeRotFromX(GetVelocity());
	NewRotator.Yaw = NewRotator.Yaw - 90; // Proper orientation :)
	SetActorRotation(NewRotator);
}
	
// Called to bind functionality to input
void ANLPBot::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

void ANLPBot::TakeUserInput(FString UserInput) {
	if (bIsBotThinking)
	{
		return;
	}
	
	bIsBotThinking = true;
	MakeRequest(UserInput);
}

void ANLPBot::MakeRequest(FString InputText)
{
	FHttpModule* Http = &FHttpModule::Get();

	TSharedRef<IHttpRequest> Request = Http->CreateRequest();
	Request->OnProcessRequestComplete().BindUObject(this, &ANLPBot::OnResponseReceived);
	
	//This is the url on which to process the request
	Request->SetURL("http://127.0.0.1:5000/inference?text=" + FGenericPlatformHttp::UrlEncode(InputText));
	Request->SetVerb("GET");
	Request->SetHeader(TEXT("User-Agent"), "X-UnrealEngine-Agent");

	Request->ProcessRequest();
}

void ANLPBot::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful)
{
	if (bWasSuccessful)
	{
		const FString ResponseString = Response->GetContentAsString();

		TSharedPtr<FJsonObject> JsonParsed;
		const TSharedRef<TJsonReader<TCHAR>> JsonReader = TJsonReaderFactory<TCHAR>::Create(ResponseString);
		if (FJsonSerializer::Deserialize(JsonReader, JsonParsed))
		{
			DoAction(JsonParsed);
		}
		else
		{
			SendMessageToChat("Didn't catch that!");
		}
	} else
	{
		SendMessageToChat("Didn't catch that!");
	}

	// Bot ended the requested action
	bIsBotThinking = false;
}

void ANLPBot::SendLocationsToServer()
{
	// Get All Locations
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), ALocation::StaticClass(), Locations);

	// Create the json with Location Titles
	FString TitlesListAsString = "";
	for (AActor* Actor : Locations)
	{
		ALocation* LocationActor = Cast<ALocation>(Actor);
		if (!LocationActor)
		{
			continue;
		}

		FString LocationTitle = LocationActor->GetTitle();
		if (TitlesListAsString.IsEmpty())
		{
			TitlesListAsString = "\"" + LocationTitle + "\"";
		} else
		{
			TitlesListAsString += ", \"" + LocationTitle + "\"";
		}

	}

	const FString ContentToSend = "{\"locations\": [" + TitlesListAsString + "]}";
	
	// Create and send request
	TSharedRef<IHttpRequest> HttpRequest = FHttpModule::Get().CreateRequest();
	HttpRequest->SetVerb("POST");
	HttpRequest->SetHeader("Content-Type", "application/json");
	HttpRequest->SetURL("http://127.0.0.1:5000/locations");
	HttpRequest->SetContentAsString(ContentToSend);

	HttpRequest->ProcessRequest();
}

void ANLPBot::SendMessageToChat(FString Message)
{
	APlayerController* PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);
	ADisertationPlayerController* DisertationPlayerController = Cast<ADisertationPlayerController>(PlayerController);
	if (DisertationPlayerController)
	{
		DisertationPlayerController->BotWriteToChatEvent(Message);
	}
}

// When answering to a question just write to chat the answer
// If answer doesn't exist just write something there.
void ANLPBot::DoAnswerQuestionAction(TSharedPtr<FJsonObject> JsonObject)
{
	FString ChatAnswer = JsonObject->GetStringField("answer");
	if (ChatAnswer.IsEmpty())
	{
		ChatAnswer = "Didn't catch that!";
	}
	
	SendMessageToChat(ChatAnswer);
}

void ANLPBot::DoFollowAction(TSharedPtr<FJsonObject> JsonObject)
{
	const FString Location = JsonObject->GetStringField("location");
	if (Location.IsEmpty())
	{
		// Should not happen, but it can be possible if the server couldn't find a location. Just write to chat
		// whatever the server answered with
		SendMessageToChat(JsonObject->GetStringField("answer"));
		return;			
	}

	ALocation* LocationActor = GetLocationFromTitle(Location);
	if (!LocationActor)
	{
		SendMessageToChat("Couldn't find the location: " + Location);
		return;
	}

	// TODO: Complete this piece of code
	ADisertationAIController* AIController = Cast<ADisertationAIController>(GetController());
	if (!AIController)
	{
		SendMessageToChat("Couldn't find the location: " + Location);
		return;
	}

	const FString ChatAnswer = JsonObject->GetStringField("answer");
	SendMessageToChat(ChatAnswer);
	AIController->MoveToActor(LocationActor, 5.f);
}

void ANLPBot::DoAction(TSharedPtr<FJsonObject> JsonObject)
{
	// Take the action enum and use it to do an action
	const EActionType ActionType = GetActionFromString(JsonObject->GetStringField("action"));

	// Unknown action, nothing to do
	if (ActionType == EActionType::UNKNOWN)
	{
		return;
	}

	if (ActionType == EActionType::ANSWER_QUESTION)
	{
		DoAnswerQuestionAction(JsonObject);
	}

	if (ActionType == EActionType::FOLLOW_ACTION)
	{
		DoFollowAction(JsonObject);		
	}
}

ALocation* ANLPBot::GetLocationFromTitle(FString LocationTitle)
{
	for (AActor* Actor: Locations)
	{
		ALocation* LocationActor = Cast<ALocation>(Actor);

		if (LocationTitle.Equals(LocationActor->GetTitle()))
		{
			return LocationActor;
		}
	}

	return nullptr;
}

EActionType ANLPBot::GetActionFromString(FString ActionString)
{
	if (ActionString == "AnswerQuestion")
	{
		return EActionType::ANSWER_QUESTION;
	}

	if (ActionString == "FollowAction")
	{
		return EActionType::FOLLOW_ACTION;
	}

	return EActionType::UNKNOWN;
}

