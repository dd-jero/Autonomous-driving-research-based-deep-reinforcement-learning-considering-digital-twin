using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using Unity.MLAgents.Sensors;
// using UnityEditor.PackageManager;
using UnityEngine;

public class obs_car_1 : MonoBehaviour
{
    private Transform tr;
    private Rigidbody rb;
    private float moveSpeed = 10f;
    private float Turn = 0f;
    private Vector3 direction;

    public float dis_Finish; // 종료 거리
    private float dis_Traveled;
    Vector3 startPosition;
    Vector3 startRotation;

    int step = 0;
    //float dis_left = 0;
    float dis_right = 0;
    float targetDisRight = 1.17341f;
    float tolernace = 0.15f;

    float Kp = 3.0f; // 비례 제어 계수
    float Ki = 1.5f; // 적분 제어 계수
    float Kd = 3.0f; // 미분 제어 계수

    float integral = 0f; // 적분 값
    float previousError = 0f; // 이전 오차 값

    private RayPerceptionSensorComponent3D raySensorComponent;

    float error;
    float proportional;
    float derivative;
    float pidValue;

    public void Awake() // 에피소드가 시작되기 전에 모든 변수 초기화하는 메소드 (start보다 먼저 호출됨)
    {
        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
        //dis_Finish = 400f;       // 여기서 변수 초기화를 시켜줘야 FixedUpdate() 적용됨

        // 물리엔진 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 위치 초기화
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();    // 해당 컴포넌트를 raySensorComponent에 저장

    }

    private void FixedUpdate() // 물리 업데이트와 관련된 작업 수행 메소드 
    {

        var input = raySensorComponent.GetRayPerceptionInput(); // ray sensor로 얻는 데이터를 input에 저장
        var output = RayPerceptionSensor.Perceive(input); // ray cast (광선 투사) 결과를 output에 저장 
        step++;
        //Debug.Log("step" + step);
        for (var rayIndex = 0; rayIndex < output.RayOutputs.Length; rayIndex++)
        {
            var extent = input.RayExtents(rayIndex); // 주어진 rayindex에 의한 시작/종료 포인트을 extent에 저장 
            Vector3 startPositionWorld = extent.StartPositionWorld; // rayindex의 시작 포인트를 해당 변수에 저장
            Vector3 endPositionWorld = extent.EndPositionWorld; // rayindex의 종료 포인트를 해당 변수에 저장

            var rayOutput = output.RayOutputs[rayIndex];

            if (rayOutput.HasHit) // ray가 부딪혔을 때 거리 계산
            {
                // Lerp()로 spw + (epw-spw)*ht 이용한 보간 (두 점 사이의 선을 따라 어느 정도 지점을 찾는데 사용)
                Vector3 hitposition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);    // hitFraction은 적중 개체까지의 정규화된 거리 
                //Debug.DrawLine(startPositionWorld, hitposition, Color.red);

                if (rayIndex == 1)
                {
                    dis_right = Vector3.Distance(hitposition, startPositionWorld); // Vector3.Distance는 인자간 거리를 반환
                    //Debug.Log("dis_right :  " + Vector3.Distance(hitposition, startPositionWorld));
                }

            }

        }

        error = targetDisRight - dis_right;

        // PID 제어 계산
        proportional = error;
        integral += error * Time.deltaTime;
        derivative = (error - previousError) / Time.deltaTime;

        pidValue = Kp * proportional + Ki * integral + Kd * derivative;

        if (Mathf.Abs(pidValue) < tolernace)
        {
            Turn = 0;
            moveSpeed = 10.0f;
        }
        else
        {
            Turn = Mathf.Clamp(-pidValue, -30f, 30f);
            moveSpeed = 5.5f;
        }

        tr.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward); // Translate는 객체를 이동시키는 메소드
        tr.Rotate(new Vector3(0f, Turn, 0f));    // Rotate는 객체를 회전시키는 메소드 

        dis_Traveled = Vector3.Distance(tr.position, startPosition);
        //Debug.Log("dis_Traveled : " + dis_Traveled);

    }

    private void OnTriggerEnter(Collider other) // 충돌 시 실행되는 메소드 
    {
        if (other.CompareTag("obstacle")) // 장애물 차량과 충돌 시
        {
            // 물리엔진 초기화
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            // 위치 초기화
            tr.position = startPosition;
            tr.eulerAngles = startRotation;

        }
    }

}
