// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Location.generated.h"

UCLASS()
class DISERTATION_API ALocation : public AActor
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, Category = "location", meta = (AllowPrivateAccess = "true"))
	FString Title;
	UPROPERTY(EditAnywhere, Category = "location", meta = (AllowPrivateAccess = "true"))
	FString Description;
	
public:	
	// Sets default values for this actor's properties
	ALocation();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	
	UFUNCTION(BlueprintPure)
	FORCEINLINE FString GetTitle() { return Title; }
	UFUNCTION(BlueprintPure)
	FORCEINLINE FString GetDescription() { return Description; }
	UFUNCTION(BlueprintCallable)
	void SetTitle(FString NewTitle);
	UFUNCTION(BlueprintCallable)
	void SetDescription(FString NewDescription);
};
