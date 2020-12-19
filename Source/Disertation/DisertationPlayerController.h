// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "DisertationPlayerController.generated.h"

/**
 * 
 */
UCLASS(Abstract)
class DISERTATION_API ADisertationPlayerController : public APlayerController
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable)
	void BotWriteToChatEvent(const FString& BotAnswer);
	
};
