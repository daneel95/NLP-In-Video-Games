// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Runtime/Online/HTTP/Public/Http.h"
#include "NLPBot.generated.h"


UENUM()
enum class EActionType : uint8
{
	ANSWER_QUESTION = 0,
    FOLLOW_ACTION = 1,
    UNKNOWN = 2
};


UCLASS()
class DISERTATION_API ANLPBot : public ACharacter
{
	GENERATED_BODY()

private:
	FString UserInputText;
	bool bIsBotThinking;

	// Actor locations -- used to know where to go in case of follow action
	UPROPERTY()
	TArray<AActor*> Locations;

	void SendLocationsToServer();
	void SendMessageToChat(FString Message);
	void DoAction(TSharedPtr<FJsonObject> JsonObject);
	EActionType GetActionFromString(FString ActionString);
	class ALocation* GetLocationFromTitle(FString LocationTitle);

	void DoAnswerQuestionAction(TSharedPtr<FJsonObject> JsonObject);
	void DoFollowAction(TSharedPtr<FJsonObject> JsonObject);

public:
	// Sets default values for this character's properties
	ANLPBot();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	// Called every frame
	virtual void Tick(float DeltaTime) override;

public:	
	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	// Take user input
	UFUNCTION(BlueprintCallable)
	void TakeUserInput(FString UserInput);

	// Make request to flask server
	UFUNCTION(BlueprintCallable)
	void MakeRequest(FString InputText);

	void OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful);
};